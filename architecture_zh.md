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
- `qdte.max_iters: 5000`
- `qdte.total_candidates_per_iter: 4096`
- `qdte.accepted_per_iter: 64`

这是当前质量优先的默认基线配置。

### `configs/adult_qdte_gpu_highpower.yaml`

高吞吐 GPU 实验配置。

关键配置：

- 输出目录：`outputs/adult_qdte_gpu_highpower`
- `qdte.candidate_backend: jax_repair`
- `qdte.transport_delta_backend: jax_prefix`
- `qdte.transport_prefix_strategy: best_advantage`
- `qdte.max_iters: 500`
- `qdte.total_candidates_per_iter: 131072`
- `qdte.accepted_per_iter: 512`
- `qdte.gpu_source_draws: 32`
- `qdte.gpu_return_top_k: 4096`

这一路径会在 GPU 上生成并评分更多候选，然后只把 top-k 候选返回到 CPU 侧做非冲突选择和 transport。

### 其他配置

- `configs/smoke.yaml`：小规模 smoke 测试配置。
- `configs/acs_qdte.yaml`：ACS 数据集配置。

## 顶层包结构

```text
qdte/
  config.py                 YAML 读取、命令行 dotted-key override
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
  measurement/
    measure.py              真实数据 measurement、DP 加噪、projection
    projection.py           simplex projection 和 count clipping
  evolution/
    engine.py               端到端 QDTE 主控流程
    state.py                QDTE 可变状态 dataclass
    initialization.py       independent one-way 初始化
    scheduler.py            active query selection
    candidates.py           CPU repair candidate generation
    gpu_candidates.py       JAX/GPU fused candidate generation and scoring
    scoring.py              JAX dense candidate scoring 和 delta 计算
    transport.py            nonconflicting selection 和 batch transport
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
   - 根据配置对 noisy counts 做 projection 或 clipping。
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

这种数组布局很重要，因为可以直接传入 JAX kernel。

### Workload groups

`build_workload` 构造 `QueryCatalogue` 和 `WorkloadGroup` 列表。

已实现的 query families：

- `oneway`
- `twoway`
- `prefix`
- `range`
- `mixed`

每个 `WorkloadGroup` 包含：

- `query_indices`
- `family`
- `sensitivity_l2`
- `is_partition`

one-way 和部分 two-way 这类 partition groups 可以投影到 simplex，使非负 counts 之和等于真实数据行数。

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

`measure_real_dataset` 返回 `Measurements` dataclass：

- `target_noisy`：projection 之前的 noisy answers。
- `target_projected`：QDTE 使用的 projected/clipped targets。
- `variances`：每个 query 的 noise variance。
- `inv_variances`：weighted loss 使用的 inverse variance。
- `groups`：每个 measurement group 的预算、noise、sensitivity 元数据。
- `mode`、`rho_total`、`epsilon_delta`、`delta`。

projection helpers：

- `project_simplex`：把 partition marginal 投影为非负且总和为 dataset size 的 counts。
- `clip_counts`：把非 partition counts clip 到 `[0, N]`。

## QDTE 状态

实现文件：`qdte/evolution/state.py`。

`QDTEState` 包含：

- `X_syn`：当前 encoded synthetic table。
- `answer_syn`：当前 synthetic query answers。
- `target`：DP noisy/projected target answers。
- `residual`：`target - answer_syn`。
- `variance`、`inv_variance`、`sigma`。
- `debt`：预留的 scheduling signal。
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

### GPU repair backend：`qdte/evolution/gpu_candidates.py`

通过以下配置启用：

```yaml
qdte:
  candidate_backend: jax_repair
```

这个 backend 把 candidate generation 和 dense scoring 融合到 JAX `pmap` kernel 中：

- 用 `replicate_table_to_devices` 把 `X_syn` 复制到本地每张 GPU。
- 在 GPU 上采样 source rows。
- 用 `_eval_candidate_source_satisfaction` 做 source satisfaction filtering。
- 用 `_repair_directed_rows` 执行 directed enter/exit repairs。
- 用 `_random_mutation_rows` 执行 random mutations。
- 在 GPU 上计算 edit costs。
- 在 GPU 上计算所有 query deltas。
- 在 GPU 上给所有 generated candidates 打分。
- 可选地通过 `gpu_return_top_k` 只返回 local top-k candidates。

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
- `compute_deltas`：为 selected candidates 计算完整 query deltas。
- `edit_advantage_from_delta`：测试/辅助函数，用于校验 advantage 计算。

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
5. 用 `select_top_nonconflicting` 选择 candidate edits。
6. 用 `choose_transport_batch` 或 `choose_transport_batch_jax` 选择 transport batch。
7. 应用 accepted edits。
8. 增量更新：

```text
answer_syn = answer_syn + transport.delta_sum
residual = target - answer_syn
```

9. 周期性调用 `answer_queries(state.X_syn, qcat)` 做 full recompute，消除或检测 incremental drift。
10. 按 `log_every` 和特殊条件记录 timeseries metrics。

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
- `metrics_timeseries.csv`：按 log interval 记录的优化指标。
- `runtime.json`：wall time、阶段耗时、throughput counters、backend 名称。
- `logs.txt`：运行日志。

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
- `tests/test_repairs.py`：candidate repair 行为。

最近一次验证的测试结果：

```text
13 passed
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

观测结果：

- `final_measured_loss`：约 `5800.76`
- `true_query_mae`：约 `0.000388`
- `true_query_rmse`：约 `0.001548`
- `num_candidates_scored`：`65,536,000`
- 运行期间观测到双卡忙时总功耗均值约 `388W`，峰值约 `438W`。

这个配置更快，也更充分地使用 GPU；但默认 5000-step 配置目前仍有更好的最终质量。

## 重要实现备注

- DP 路径不是模拟实验。它会在真实输入表上计算 query answers，按 zCDP Gaussian mechanism 加噪，做 projection/clipping，然后 QDTE 针对这些 noisy targets 优化。
- Oracle mode 被显式隔离，并会打印 warning。
- 默认路径优先保证最终质量和可靠性。
- 高吞吐路径优先提高 GPU occupancy 和候选评分吞吐。
- 当前只实现了 `measurement_mode=static_all`。adaptive select-measure-generate 在这个版本中会被 `NotImplementedError` 明确拒绝。
- 配置里有 `include_halfspace` 兼容字段，但 `build_workload` 目前没有实现 halfspace workload construction。
- dense query evaluation 使用 `(batch_size, num_queries)` 的 boolean satisfaction matrix。这种实现简单且适合 GPU，但可能受 memory bandwidth 限制。
