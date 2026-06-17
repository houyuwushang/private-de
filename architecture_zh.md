# QDTE 实现架构说明

本文档说明当前仓库中 QDTE 合成数据生成器的代码结构、执行路径、隐私边界和 GPU 优化实现。目标是让代码审查时可以直接从文档定位到对应文件，而不是只看抽象描述。

## 入口脚本

### `scripts/run_qdte.py`

主命令行入口。

职责：

- 用 `qdte.config.load_yaml` 读取 YAML 配置。
- 支持命令行 dotted-key 覆盖，例如 `--privacy.mode dp`。
- 根据 `runtime.xla_preallocate` 控制 XLA 是否预分配显存。
- 调用 `qdte.evolution.engine.run_qdte(config)` 执行完整流程。

主要 DP 运行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

### `scripts/smoke_qdte.py`

小规模 smoke 测试入口。它会在 `outputs/` 下生成一个 toy CSV，然后构造内存配置并调用 `run_qdte`。

### `scripts/run_ablation.py`

消融实验入口，用于比较不同变体：

- `random_mutation`
- `no_edit_cost`
- `sequential_greedy`
- `target_only`
- `no_threshold`

### `scripts/check_env.py`

JAX/GPU 环境检查脚本。它会打印 JAX 版本、设备列表、本地设备数量，并运行一个矩阵乘法。

## 配置文件

### `configs/adult_qdte.yaml`

Adult 数据集默认可靠 DP 配置。

关键配置：

- 输入 CSV：`/home/qianqiu/rerun-experiment/dataset/adult.csv`
- 输出目录：`outputs/adult_qdte`
- `privacy.mode: dp`
- `privacy.rho_total: 1.0`
- `privacy.delta: 1.0e-9`
- `privacy.measurement_mode: static_all`
- `qdte.candidate_backend` 未显式设置，因此 engine 默认使用 `cpu_repair`
- `qdte.score_backend: dense_gpu`
- `qdte.transport_mode: atom_flow`
- `qdte.atom_flow_update_mode: batch`
- `qdte.atom_flow_pool_multiplier: 16`
- `qdte.atom_flow_max_pool: 0`
- `qdte.max_iters: 5000`
- `qdte.total_candidates_per_iter: 4096`
- `qdte.accepted_per_iter: 64`

这是当前质量优先的默认基线配置。

### `configs/adult_qdte_gpu_highpower.yaml`

高吞吐 GPU 实验配置。

关键配置：

- 输出目录：`outputs/adult_qdte_gpu_highpower`
- `qdte.candidate_backend: jax_repair`
- `qdte.score_backend: sparse_delta_gpu`
- `qdte.transport_mode: atom_flow`
- `qdte.atom_flow_update_mode: batch`
- `qdte.transport_delta_backend: jax_prefix`
- `qdte.transport_prefix_strategy: best_advantage`
- `qdte.max_iters: 500`
- `qdte.total_candidates_per_iter: 1572864`
- `qdte.accepted_per_iter: 1024`
- `qdte.gpu_source_draws: 32`
- `qdte.gpu_batches_per_iter: 1`
- `qdte.gpu_sparse_query_block_size: 64`
- `qdte.gpu_sparse_changed_attr_capacity: 4`
- `qdte.gpu_return_top_k: 8192`

这一路径会在 GPU 上生成并评分更多候选。默认 scorer 使用 sparse-delta GPU path，只对每个 edit 可能影响的 query blocks 做 predicate evaluation，然后只把 top-k 候选返回到 CPU 侧做 atom-flow transport。

### 其他配置

- `configs/smoke.yaml`：小规模 smoke 测试配置。
- `configs/acs_qdte.yaml`：ACS 数据集配置。

## 顶层包结构

```text
qdte/
  config.py                 YAML 读取、命令行 dotted-key override
  config_validation.py      unsupported mode/backend 的 fail-fast 校验
  dataio.py                 输出目录、JSON、NPY 辅助函数
  preprocess.py             CSV 编码和合成数据解码
  schema.py                 编码表 schema dataclass
  privacy/
    accountant.py           zCDP epsilon 换算
    gaussian.py             zCDP Gaussian mechanism
    exponential.py          exponential mechanism 辅助函数，默认 QDTE 路径不使用
  queries/
    types.py                query catalogue 表示
    workload.py             workload 和 measurement group 构造
    eval_jax.py             JAX query evaluation
    delta_index.py          edit delta 用的 sparse affected-query index
  measurement/
    measure.py              真实数据 measurement、DP 加噪、projection
    projection.py           simplex projection 和 count clipping
    consistency.py          scope-local marginal consistency projection
  evolution/
    engine.py               端到端 QDTE 主控流程
    state.py                QDTE 可变状态 dataclass
    initialization.py       independent one-way 初始化
    scheduler.py            active query selection
    candidates.py           CPU repair candidate generation
    gpu_candidates.py       JAX/GPU fused candidate generation and scoring
    scoring.py              dense/sparse candidate scoring 和 delta 计算
    transport.py            nonconflicting、prefix 和 atom-flow transport
  eval/
    metrics.py              measured loss 和 true-query evaluation metrics
    runtime.py              runtime counters 和 throughput stats
```

## 端到端执行流程

完整流程由 `qdte.evolution.engine.run_qdte` 组织。

高层步骤：

1. 解析配置和输出目录。
2. 读取并编码真实 CSV。
3. 构建 query workload。
4. 对真实数据做 measurement：
   - 在 `X_real` 上计算真实 query answers；
   - 如果 `privacy.mode=dp`，按 zCDP Gaussian mechanism 加噪；
   - 根据配置对 noisy counts 做 simplex、prefix monotonicity、或 scope-local marginal consistency projection。
5. 用 one-way noisy targets 初始化合成数据。
6. 运行 QDTE edit loop：
   - 选择 active target queries；
   - 生成候选 edits；
   - 针对 measured target 给候选 edits 打分；
   - 选择非冲突 edits；
   - 选择正收益 transport prefix；
   - 应用 accepted edits；
   - 增量更新 synthetic query answers；
   - 周期性全量重算，检查 incremental drift。
7. 保存 encoded 和 decoded synthetic data。
8. 保存最终指标、timeseries、runtime profile、schema、workload、measurements、logs 和 resolved config。

## 隐私边界

隐私关键代码在 `qdte.measurement.measure.measure_real_dataset`。

### DP mode

当 `privacy.mode=dp` 时：

1. `answer_queries(X_real, qcat, batch_size=...)` 在真实编码数据上计算 exact query answers。
2. `_allocate_group_budgets` 给 workload groups 分配 zCDP 预算。
3. 每个 workload group 调用 `qdte/privacy/gaussian.py` 中的 `add_zcdp_gaussian_noise`。
4. 加噪结果写入 `Measurements.target_noisy`。
5. projection/clipping 后的 noisy counts 写入 `Measurements.target_projected`。
6. QDTE 只针对 `target_projected` 优化，并使用 Gaussian noise variance 产生的 inverse variance 作为 loss 权重。

Gaussian mechanism 使用：

```text
sigma = 1 / sqrt(2 * rho)
noise_std = sensitivity_l2 * sigma
noisy_answers = true_answers + Normal(0, noise_std)
```

报告的 epsilon 在 `qdte/privacy/accountant.py` 中计算：

```text
epsilon(delta) = rho + 2 * sqrt(rho * log(1 / delta))
```

### Oracle mode

当 `privacy.mode=oracle` 时，`measure_real_dataset` 会直接使用 exact real query answers 作为 target，并打印 warning。

这个模式只用于 debug 或 upper-bound 实验，不是 DP 结果路径。

### 生成后的 evaluation

`evaluation.compute_true_query_error` 会在运行结束后计算 exact true query answers，用于报告最终 MAE/RMSE。这些 exact answers 在 DP mode 下不会作为 QDTE 优化目标使用。

## 数据编码

实现文件：

- `qdte/preprocess.py`
- `qdte/schema.py`

### `load_and_preprocess_csv`

读取配置里的 CSV，并返回：

- `PreprocessResult.X`：整数编码表，shape 为 `(n_rows, n_columns)`。
- `PreprocessResult.schema`：包含列元数据的 `TableSchema`。
- `PreprocessResult.raw_columns`：原 CSV 列名，统一转成字符串。

categorical columns 会映射到排序后的整数 category IDs。numerical columns 会离散化成近似分位数 bins，除非 unique value 数量已经很小。missing values、空字符串和 `?` 会映射到配置的 missing token。

### `decode_array`

用 schema representatives 把 encoded synthetic records 转回可以写 CSV 的 `pandas.DataFrame`。

## Query 表示和 Workload

实现文件：

- `qdte/queries/types.py`
- `qdte/queries/workload.py`
- `qdte/queries/eval_jax.py`

### Query terms

支持的操作：

- `OP_EQ`
- `OP_LE`
- `OP_GE`
- `OP_RANGE`

每个 query 存储在固定大小数组中：

- `attrs`
- `ops`
- `values`
- `lows`
- `highs`
- `num_terms`
- `linear_attrs`
- `linear_weights`
- `linear_thresholds`
- `linear_num_terms`

普通数组表示 `EQ/LE/GE/RANGE` conjunction。linear 数组表示 halfspace predicate：

```text
sum_i weight_i * x[attr_i] <= threshold
```

这种固定数组布局很重要，因为可以直接传入 JAX kernel。

### Workload groups

`build_workload` 构造 `QueryCatalogue` 和 `WorkloadGroup` 列表。

已实现的 query families：

- `oneway`
- `twoway`
- `prefix`
- `range`
- `mixed`
- `halfspace`

每个 `WorkloadGroup` 包含：

- `query_indices`
- `family`
- `sensitivity_l2`
- `is_partition`

one-way 和部分 two-way 这类 partition groups 可以投影到 simplex，使非负 counts 之和等于真实数据行数。

Halfspace 查询已支持 measurement、evaluation、CPU repair 和 consistency projection。GPU fused candidate repair 会对 `include_halfspace: true` fail-fast，因为该 backend 还没有实现 directed halfspace repair。

### JAX query evaluation

`qdte/queries/eval_jax.py` 提供：

- `eval_records_queries_arrays`：JIT 编译的 per-record/per-query boolean matrix evaluation。
- `eval_records_queries`：使用 `QueryCatalogue` 的 wrapper。
- `answer_queries`：按 record batch 计算并累加 query satisfaction indicators。

核心 count 公式：

```text
answer[q] = sum_i 1{record_i satisfies query_q}
```

## Measurement 和 Projection

实现文件：

- `qdte/measurement/measure.py`
- `qdte/measurement/projection.py`
- `qdte/measurement/consistency.py`

`measure_real_dataset` 返回 `Measurements` dataclass：

- `target_noisy`：projection 之前的 noisy answers。
- `target_projected`：QDTE 使用的 projected/clipped targets。
- `variances`：每个 query 的 noise variance。
- `inv_variances`：weighted loss 使用的 inverse variance。
- `groups`：每个 measurement group 的预算、noise、sensitivity 元数据。
- `mode`、`rho_total`、`rho_spent`、`epsilon_delta`、`delta`。
- `projection_diagnostics`：projection 元数据，包括 consistency diagnostics。

projection helpers：

- `project_simplex`：把 partition marginal 投影为非负且总和为 dataset size 的 counts。
- `clip_counts`：把非 partition counts clip 到 `[0, N]`。
- `project_non_decreasing`：对 prefix counts 做 weighted PAVA 单调投影。

consistency projection：

- `project_consistent_targets` 把每个 query 的 attribute set 视为一个 scope。
- 对每个 scope 建立一个局部 marginal table `mu_scope >= 0`，并强制：

```text
sum_cells mu_scope = N
```

其中 `N` 是已知真实数据行数，不是 noisy measurement。

- 任意 `EQ/LE/GE/RANGE` conjunction 都会被映射成 scope table 上一组 cells 的线性和：

```text
answer(q) = sum_{cell satisfies q} mu_scope[cell]
```

- 对重叠 scopes 做 shared marginal reconciliation：

```text
marginal(mu_S over S∩T) = marginal(mu_T over S∩T)
```

这覆盖当前已实现的 `oneway`、`twoway`、`prefix`、`range`、`mixed`、`halfspace` query，以及 adaptive 外循环未来加入的任意维 supported conjunction。若某个 scope 的 cell 数超过 `projection.consistency.max_scope_cells`，代码会 fail-fast，而不是静默跳过一致性投影。

当前 projection 后仍保留 diagonal variance 表示；完整的 post-projection covariance propagation 还没有实现。

## QDTE 状态

实现文件：`qdte/evolution/state.py`。

`QDTEState` 包含：

- `X_syn`：当前 encoded synthetic table。
- `answer_syn`：当前 synthetic query answers。
- `target`：DP noisy/projected target answers。
- `residual`：`target - answer_syn`。
- `variance`、`inv_variance`、`sigma`。
- `debt`：accepted edits 后按 query-level damage/improvement 更新的 scheduling signal。
- `iteration`。

主 loss 在 `qdte/eval/metrics.py` 中实现：

```text
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
```

## 初始化

实现文件：`qdte/evolution/initialization.py`。

`initialize_independent_oneway` 会尽量从 noisy/projected one-way marginals 中独立采样每一列，构造初始 synthetic table。

如果某列没有可用的一维 target，或者 target 不合法，则对该列 cardinality 做 uniform sampling。

## Active Query Scheduling

实现文件：`qdte/evolution/scheduler.py`。

`select_active_queries` 按 standardized residual 排序：

```text
priority = (abs(residual) - kappa_noise * sigma) / sigma
```

优先选择正 priority 的 queries。如果没有正 priority，则退化为按 absolute standardized residual 选择。最终返回 top `num_active_targets` queries，用于 directed candidate repair。

## Candidate Generation

候选 edits 由 `qdte/evolution/candidates.py` 中的 `CandidateBatch` 表示：

- `row_ids`：要编辑的 synthetic row IDs。
- `old_rows`：edit 前的 rows。
- `new_rows`：edit 后的 proposed rows。
- `target_query_ids`：每个候选对应的目标 query，随机候选为 `-1`。
- `edit_cost`：Hamming/numerical edit penalty。
- `repair_type`：random、enter 或 exit repair type。
- `diagnostics`：requested/produced candidates 和 source filtering 诊断信息。

### CPU repair backend：`qdte/evolution/candidates.py`

这是默认 backend。

对每个 active query：

1. 根据 residual 判断需要更多 records 满足该 query，即 `enter`，还是更少 records 满足该 query，即 `exit`。
2. 从 `X_syn` 中采样 source rows。
3. 过滤 source rows：
   - enter repair 从当前不满足该 query 的 rows 开始；
   - exit repair 从当前满足该 query 的 rows 开始。
4. 执行 batched repair：
   - `_repair_enter_batch`：修改 query terms，使 new row 满足 target query。
   - `_repair_exit_batch`：尽量破坏至少一个 query term。
5. 按 `random_candidate_fraction` 添加 random mutation candidates。
6. 用 `compute_edit_cost` 计算 edit cost。

可选的 `qdte.candidate_compiler: masked_single_query` 是默认 one-query repair 的 target-preserving mask 版本。对于正 residual，它采样满足未 mask terms 但不满足完整 query 的 near-miss rows，然后只修复被 mask 的 terms。对于负 residual，它从满足完整 query 的 rows 出发，只破坏被 mask 的 terms，保留候选时保证退出 target query。最终接受规则仍然使用完整 measured-objective edit advantage。

可选的 `qdte.candidate_compiler: paired_query` 会先生成 paired CPU proposals，再回退到普通 one-query repair。它从 active set 中选择负 residual source query 和正 residual destination query，采样满足 source 且不满足 destination 的旧记录，把它们 repair 到 destination，并可选地尝试在保持 destination membership 的同时破坏 source membership。`qdte.candidate_compiler: masked_paired_query` 会用普通 query terms 的随机子集作为 source/destination mask，扩大 proposal 搜索空间，但最终仍使用完整 measured objective 评分。`qdte.candidate_compiler: masked_exit_query` 会采样已经满足正 residual destination mask 的行，只对负 residual source mask 做 exit，从而保留 destination mask。这些 paired proposals 不会因为匹配 pair 或 mask 就直接接受。

实验性的 exit-axis compilers 用来拆分 candidate generation 的设计轴，同时保持 scoring 和 transport objective 不变。`directed_exit_only` 只使用负 residual queries，采样满足该 query 的旧记录，并编译 directed exit repair。`masked_exit_only` 是 single-query mask 版本：从负 residual query 的普通 terms 中采样子集，只让记录退出这个 mask，最后仍用完整 measured objective 评分。`random_source_directed_exit` 保留负 residual query 的 edit 方向，但移除 source filtering，让旧记录来源随机。这些 variant 都是 CPU-only ablation，用来在同一个 QDTE scorer 下比较 QDTE-style 定向 proposal 和 Private-GSD-style random mutation。

实验性的 `qdte.transport_mode: blind_accept` 会绕过 positive-advantage 和 batch/prefix objective gate，按生成顺序接受不冲突的候选。它只用于 sanity-check 消融；它不是 Private-GSD 的等价实现，因为 Private-GSD 仍然会按整张候选数据集的 fitness 排名。

### GPU repair backend：`qdte/evolution/gpu_candidates.py`

通过以下配置启用：

```yaml
qdte:
  candidate_backend: jax_repair
```

这个 backend 把 candidate generation 和 scoring 融合到 JAX `pmap` kernel 中：

- 用 `replicate_table_to_devices` 把 `X_syn` 复制到本地每张 GPU。
- 用 cached GPU context 复用 query/schema/device 常量，避免每轮重复 device-put。
- 对 active query ids 做固定容量 padding，减少 active query 数变化导致的 `pmap` recompilation。
- 在 GPU 上采样 source rows。
- 用 `_eval_candidate_source_satisfaction` 做 source satisfaction filtering。
- 用 `_repair_directed_rows` 执行 directed enter/exit repairs。
- 用 `_random_mutation_rows` 执行 random mutations。
- 在 GPU 上计算 edit costs。
- `score_backend: dense_gpu` 时，在 GPU 上用 boolean enter/exit masks 计算等价的 dense query delta 贡献，减少 float32 delta 临时张量压力。
- dense path 可选地按 `gpu_score_query_block_size` 对 query 维度分块累加 score，避免一次性形成完整的 candidate x query 临时矩阵。
- `score_backend: sparse_delta_gpu` 时，预先构造 attribute-to-query affected index 和 multi-word query scope bitsets；每个 edit 只遍历 changed attrs 对应的 query blocks，并用 bitset 去重同一 query 被多个 changed attrs 重复计入的问题。
- sparse GPU path 支持 `QueryCatalogue` 中普通 k-way `EQ/LE/GE/RANGE` conjunction；高吞吐 GPU repair 仍不支持 directed halfspace repair，因此 halfspace workload 搭配 GPU candidate backend 会 fail-fast。
- 在 GPU 上给所有 generated candidates 打分。
- 可选地通过 `gpu_return_top_k` 只返回 local top-k candidates。
- 可选地通过 `gpu_batches_per_iter` 在同一个 QDTE iteration 内运行多个 GPU scoring batch。

即使只返回 top-k，`CandidateBatch.diagnostics["scored_candidates"]` 也会记录实际在 GPU 上评分的完整 candidate 数量。

当 GPU candidate backend 启用时，accepted edits 之后会通过 `apply_edits_to_replicated_table` 同步更新 replicated GPU table。

## Candidate Scoring

实现文件：`qdte/evolution/scoring.py`。

对于候选 edit `x_old -> x_new`，定义：

```text
delta[q] = 1{x_new satisfies q} - 1{x_old satisfies q}
weights[q] = residual[q] * inv_variance[q]
```

score 公式：

```text
advantage =
    delta dot weights
    - 0.5 * sum_q delta[q]^2 * inv_variance[q]
    - lambda_cost * edit_cost
```

这表示在 weighted quadratic objective 下，单个 edit 的目标函数改进量，并包含 edit penalty。

可用 scoring paths：

- `score_candidates`：对所有 queries 做 dense JAX scoring，可选 multi-GPU `pmap`。
- `score_candidates_target_only`：较便宜的消融路径，只按候选对应 target query 打分。
- `score_candidates_sparse`：基于 `QueryDeltaIndex` 的 exact CPU sparse-delta scoring，适合 audit/debug。
- `sparse_delta_gpu`：GPU fused candidate backend 内部的 compiled sparse-delta scoring。
- `compute_deltas`：为 selected candidates 计算完整 query deltas。
- `compute_deltas_sparse`：从 sparse affected-query evaluation 生成 dense delta matrix。
- `edit_advantage_from_delta`：测试/辅助函数，用于校验 advantage 计算。

实现上，dense JAX scoring 使用 `enter = phi_new & ~phi_old` 和 `exit = phi_old & ~phi_new` 计算：

```text
linear = enter dot weights - exit dot weights
quad = (enter or exit) dot inv_variance
```

这与显式构造 `delta = phi_new - phi_old` 后计算 `delta dot weights` 和 `delta^2 dot inv_variance` 等价，但减少了大批量 GPU scoring 的临时 float32 tensor 压力。

高功率 dense path 还可以通过 `gpu_score_query_block_size` 把 query 维度分块，逐块累加同一个 `linear` 和 `quad`。这不会改变 score 数学定义，只改变 GPU kernel 的峰值临时内存布局。

当前 highpower 默认使用 `sparse_delta_gpu`。它把 `QueryDeltaIndex` 编译成 padded `attr -> qids` 表，并为每个 query 存 multi-word `uint32` attribute scope bitsets。GPU kernel 对每个 candidate 的 changed attrs 做固定容量遍历，只评估这些 attrs 可能影响的 query blocks；如果一个 query 同时包含多个 changed attrs，scope bitsets 用来保证该 query 只在最小 changed attr 上计入一次。该路径仍计算同一个 edit advantage 公式，不改变 QDTE objective。

multi-word bitsets 已移除旧版单 `uint32` 的 31 encoded-attribute 限制。剩余容量假设是 `gpu_sparse_changed_attr_capacity`：它必须覆盖一个候选 edit 可能修改的属性数。当前 Adult highpower 配置把该值设为 `4`，与 `workload.max_terms: 4` 对齐。

## Selection 和 Transport

实现文件：`qdte/evolution/transport.py`。

### 非冲突选择

`select_top_nonconflicting`：

1. 保留 finite 且 advantage 大于 `min_advantage` 的候选。
2. 按 score 排序候选池。
3. 每个 synthetic row 最多接受一个 candidate。
4. 返回最多 `accepted_per_iter` 个 candidate indices。

这样可以避免同一个 microbatch 内多个 edits 试图修改同一行 synthetic row。

### CPU transport prefix

`choose_transport_batch`：

1. 按 individual advantage 对 selected candidates 排序。
2. 计算 cumulative query delta prefixes。
3. 为每个 prefix 计算真实 batch advantage。
4. 接受：
   - 最大正收益 prefix；或
   - 当 `transport_prefix_strategy: best_advantage` 时，接受收益最高的正 prefix。

batch advantage 会在完整 residual 上重新计算，而不是只依赖 per-candidate score。因此它会考虑多个 edits 之间的 quadratic interaction。

### JAX transport prefix

`choose_transport_batch_jax`：

通过以下配置启用：

```yaml
qdte:
  transport_delta_backend: jax_prefix
```

它使用 `_choose_transport_prefix_jit` 把 prefix delta/advantage 计算搬到 JAX 中。这减少了高吞吐配置下 CPU 侧 delta 计算和数据搬运开销。

### Atom-flow transport

`transport_mode: atom_flow` 支持两个 update mode：

- `atom_flow_update_mode: batch`：默认路径。一次选择一批 flow units，最后在 engine 中一次性更新 `answer_syn` 和 `residual`。
- `atom_flow_update_mode: exact`：逐个 flow unit 更新 current weighted residual，并精确刷新受影响 edge 的 marginal。它更接近 successive atom-flow，但在大 returned candidate pool 上 CPU 开销更高。

`choose_atom_flow_transport` 是 exact mode：

1. 从 finite 且正收益的 returned candidates 构造候选池，可用 `atom_flow_pool_multiplier` 或 `atom_flow_max_pool` 控制池大小；`atom_flow_max_pool: 0` 表示不额外截断。
2. 对候选计算完整 measured-query delta。
3. 用完整 encoded `old_row` 和 `new_row` 作为 source/target atom，构造 atom-flow edge。
4. 对每条 atom edge 保留按 individual advantage 排序的可用 row units，并用 row id capacity 保证同一 synthetic row 在同一批中最多搬一次。
5. 在当前 weighted residual 上逐个流量单位做 exact marginal update：

```text
marginal(e | Delta_F) =
  delta_e @ ((residual - Delta_F) * inv_variance)
  - 0.5 * (delta_e^2 @ inv_variance)
  - lambda_cost * edit_cost_e
```

6. 每接受一个流量单位，就更新 current weighted residual，并通过 query-to-edge inverted index 精确更新所有受影响 atom edge 的 marginal。
7. 当最佳 marginal 不再超过 `min_advantage` 或达到 `accepted_per_iter` 时停止。

最终仍会用完整 batch advantage 重新校验整批：

```text
A(B) = residual @ (Delta_F * inv_variance)
       - 0.5 * (Delta_F^2 @ inv_variance)
       - lambda_cost * total_edit_cost
```

`choose_atom_flow_batch_transport` 是 batch/JAX mode：

1. 从正收益 returned candidates 构造候选池。
2. 在 JAX 上计算 pool candidates 对全部 measured queries 的 dense delta 和 `delta^2 @ inv_variance`；initial advantage 复用 engine/scoring 已返回的 candidate scores。
3. 在 CPU 侧按完整 encoded `old_row -> new_row` atom edge 分组，并对同一 edge 用固定 residual 下的 diminishing return 估计 flow capacity：

```text
marginal_k(e) = marginal_1(e) - (k - 1) * (delta_e^2 @ inv_variance)
```

4. 对候选 flow units 排序并执行 row id capacity，得到一批不冲突 edits。
5. 在 JAX 上对这批 edits 做 cumulative prefix delta，并用完整 batch objective 选择正收益 prefix：

```text
A(B) = residual @ (Delta_B * inv_variance)
       - 0.5 * (Delta_B^2 @ inv_variance)
       - lambda_cost * total_edit_cost
```

6. engine 只在接受 prefix 后做一次：

```text
answer_syn = answer_syn + transport.delta_sum
residual = target - answer_syn
```

因此 batch atom-flow 不改变 QDTE 目标函数或隐私边界，只改变同一批候选 edits 的 transport 近似求解方式。它牺牲 exact mode 的逐单位 marginal refresh，但避免每接受一个 flow unit 都更新残差，适合高吞吐 GPU candidate/scoring path。

### 应用 edits

`apply_edits` 会原地修改 CPU `X_syn`：

```text
X_syn[row_ids[accepted]] = new_rows[accepted]
```

如果 GPU candidate backend 启用，`apply_edits_to_replicated_table` 也会同步更新 replicated GPU table。

## QDTE 主循环

主循环位于 `qdte/evolution/engine.py`。

每个 iteration：

1. 设置 `state.iteration`。
2. 用 `select_active_queries` 选择 active queries。
3. 生成 candidates：
   - CPU path：`generate_candidates`。
   - GPU path：`generate_and_score_candidates_gpu`。
4. 给 candidates 打分：
   - CPU repair path 使用 `score_candidates` 或 `score_candidates_target_only`。
   - GPU repair path 直接使用 `generate_and_score_candidates_gpu` 返回的 fused scores。
5. 选择 transport：
   - `transport_mode: atom_flow` 且 `atom_flow_update_mode: batch` 时，运行 `choose_atom_flow_batch_transport`。
   - `transport_mode: atom_flow` 且 `atom_flow_update_mode: exact` 时，运行 `choose_atom_flow_transport`。
   - 否则先用 `select_top_nonconflicting` 选择候选，再用 `choose_transport_batch` 或 `choose_transport_batch_jax` 选择 prefix batch。
6. 应用 accepted edits。
7. 增量更新：

```text
answer_syn = answer_syn + transport.delta_sum
residual = target - answer_syn
```

8. 周期性调用 `answer_queries(state.X_syn, qcat)` 做 full recompute，消除或检测 incremental drift。
9. 按 `log_every` 和特殊条件记录 timeseries metrics。

最终步骤总是重新计算 synthetic query answers，并报告 `final_incremental_answer_drift`。

## 输出文件

一次正常 run 会写出：

- `config_resolved.yaml`：CLI overrides 后的配置。
- `schema.json`：编码 schema。
- `queries.json`：query catalogue。
- `measurements.json`：DP noisy targets、variances、budget metadata。
- `synthetic_encoded.npy`：encoded synthetic table。
- `synthetic_decoded.csv`：decoded synthetic table，取决于 `evaluation.save_synthetic_csv`。
- `metrics_final.json`：最终隐私参数、loss、query error 和 run summary。
- `metrics_by_family.json`：按 query family 拆分的 measured 和可选 true-query metrics。
- `metrics_timeseries.csv`：按 log interval 记录的优化指标。
- `workload_summary.json`：measured workload 摘要。
- `runtime.json`：wall time、阶段耗时、throughput counters、backend 名称。
- `logs.txt`：运行日志。

开启 held-out evaluation 时，还会写出：

- `queries_holdout.json`
- `workload_summary_holdout.json`
- `metrics_holdout.json`
- `metrics_by_family_holdout.json`

用户要求的四个核心输出文件是：

- `synthetic_decoded.csv`
- `metrics_final.json`
- `metrics_timeseries.csv`
- `runtime.json`

## Runtime 和 Metrics

实现文件：

- `qdte/eval/metrics.py`
- `qdte/eval/runtime.py`

`RuntimeStats` 记录：

- measurement time
- initialization time
- total generation time
- candidate generation time
- scoring time
- transport time
- full recompute time
- iterations 数量
- requested/scored candidates 数量
- accepted edits 数量
- source-filter diagnostics
- accepted rate
- scoring throughput

`query_error_metrics` 报告 post-run normalized query error：

- `true_query_mae`
- `true_query_rmse`
- `true_query_max_error`

这些 metrics 只用于 evaluation，对比 exact true query rates 和 synthetic query rates。

## 测试

当前测试覆盖关键模块：

- `tests/test_engine_smoke.py`：端到端 smoke execution。
- `tests/test_measurement.py`：DP measurement/projection 行为。
- `tests/test_transport.py`：batch transport 和 selection 行为。
- `tests/test_queries.py`：query evaluation/workload 行为。
- `tests/test_edit_advantage.py`：edit advantage 一致性。
- `tests/test_repairs.py`：candidate repair 行为，包括 k-way conjunction、halfspace、generated directed candidates 的 enter/exit contract，以及 paired-query homogeneous-table edge case。
- `tests/test_projection.py`：prefix monotonicity projection 行为。
- `tests/test_consistency_projection.py`：scope-local consistency projection。
- `tests/test_config_validation.py`：unsupported mode 和 backend validation。
- `tests/test_delta_index.py`：sparse affected-query delta index 和 sparse scoring。
- `tests/test_gpu_candidates.py`：GPU padding、query-block scoring 和 sparse GPU scoring helpers。
- `tests/test_scheduler.py`：strict noise thresholding 和 debt scheduling。
- `tests/test_workload.py`：workload family construction，包括 halfspace。

最近一次验证的测试结果：

```text
76 passed in 6.92s
```

## 当前已验证运行

### 默认 Adult DP run

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

输出目录：

```text
outputs/adult_qdte
```

观测结果：

- `final_measured_loss`：约 `2533.62`
- `true_query_mae`：约 `0.000214`
- `true_query_rmse`：约 `0.000657`
- `num_candidates_scored`：`20,480,000`
- `privacy_mode`：`dp`
- `epsilon_delta`：约 `10.1046`

### 高吞吐 GPU Adult DP run

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte_gpu_highpower.yaml --privacy.mode dp
```

输出目录：

```text
outputs/adult_qdte_gpu_highpower
```

50-iteration throughput probe 观测结果：

- `num_candidates_scored`：`78,643,200`
- `time_generation_seconds`：约 `19.91`
- `time_scoring_seconds`：约 `14.26`
- `candidate_scoring_throughput_per_second`：约 `5.52M`
- `candidates_scored_per_second`：约 `3.95M`
- `final_measured_loss`：约 `127,132`
- 运行期间观测到 GPU 0 约 `290-328W`，GPU 1 约 `289-309W`。

这个配置更快，也更充分地使用 GPU；但默认质量配置和长跑质量仍需按实验目标分别比较。

## 重要实现备注

- DP 路径不是模拟实验。它会在真实输入表上计算 query answers，按 zCDP Gaussian mechanism 加噪，做 projection/clipping，然后 QDTE 针对这些 noisy targets 优化。
- Oracle mode 被显式隔离，并会打印 warning。
- 默认路径优先保证最终质量和可靠性。
- 高吞吐路径优先提高 GPU occupancy 和候选评分吞吐。
- 当前只实现了 `measurement_mode=static_all`。adaptive select-measure-generate 在这个版本中会被 `NotImplementedError` 明确拒绝。
- `include_halfspace` 已支持 measurement/evaluation/CPU repair/consistency projection；GPU fused candidate repair 对 halfspace 仍未实现，会在配置验证中 fail-fast。
- consistency projection 后仍使用 diagonal variance 表示；完整 covariance propagation 还没有实现。
- `sparse_delta_gpu` 使用 multi-word `uint32` query-scope bitsets，旧版 31 encoded-attribute 限制已移除。仍需保证 `gpu_sparse_changed_attr_capacity` 覆盖候选可能修改的属性数。
- 仓库还没有 dependency manifest；当前测试和运行依赖本机 `qdte` conda 环境。
- dense query evaluation 使用 `(batch_size, num_queries)` 的 boolean satisfaction matrix。这种实现简单且适合 GPU，但可能受 memory bandwidth 限制。
