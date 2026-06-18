# private-de 代码状态说明

本仓库实现了一个面向表格数据的 QDTE 合成数据生成器。当前代码重点是：从真实 CSV 构建查询 workload，在 DP 或 oracle measurement 下得到目标统计量，然后通过 edit-based QDTE 优化生成合成表，并输出可审计的质量、运行时和 workload 报告。

## 当前能力

- 支持读取 CSV 并编码表格 schema。
- 支持 categorical 列和 numerical 离散化列。
- 支持的查询 workload family：
  - `oneway`
  - `twoway`
  - `prefix`
  - `range`
  - `mixed`
  - `kway`
  - `kway_prefix`
  - `kway_range`
  - `kway_mixed`
  - `orthogonal_kway_mixed`
  - `halfspace`
- 支持 DP measurement：
  - 使用 zCDP Gaussian mechanism 加噪。
  - 对 partition workload 可做 simplex projection。
  - 对非 partition counts 可做 clipping。
  - 按 `WorkloadGroup` 同时测量一组查询：为每个 group 分配 rho，用该 group 的 L2 sensitivity 标定 vector Gaussian noise；当 group scope 可枚举时会按查询定义精确计算单条记录最大 overlap，正交 equality workload 的 sensitivity 为 `1`，当前记录的是 diagonal variances，组内坐标噪声独立同方差。
  - 支持 scope-local marginal consistency projection：把 `EQ/LE/GE/RANGE` 的任意维 conjunction 映射到局部 marginal table，强制非负、已知总行数 `N`、以及重叠 query scope 的边际一致性。
- 支持 oracle mode，用于 debug 或上界实验，不应作为 DP 结果使用。
- 支持 QDTE edit loop：
  - active query selection
  - CPU repair candidate generation
  - optional paired-query / masked-paired / masked-exit / exit-only CPU compiler：从 residual field 生成更结构化的 transport proposals
  - dense JAX/GPU scoring
  - non-conflicting edit selection
  - prefix-greedy microbatch transport acceptance
  - exact successive atom-flow transport over encoded row atoms
  - periodic full recompute drift check
- 支持默认严格的 noise threshold：未超过 `kappa_noise * sigma` 的 residual 默认不会触发候选生成。
- 支持基础 debt scheduler 更新，用于记录 query-level collateral damage，并可通过 `debt_alpha` 影响后续 active query priority。
- 支持 prefix monotonicity projection，可对 prefix measurement group 做 weighted isotonic projection。
- 支持 halfspace 查询的 measurement、JAX/CPU evaluation、CPU repair candidate path、GPU fused single-query repair path 和 consistency projection。
- 支持 GPU-oriented candidate path：
  - `jax_repair` / `gpu_repair`
  - cached GPU query/schema context
  - fixed-shape active query padding to reduce recompiles
  - boolean enter/exit dense scoring to reduce temporary tensor pressure
  - query-block dense scoring to avoid one huge candidate x query temporary matrix
  - top-k candidates returned to CPU
  - JAX prefix transport delta path
- 支持 audit 输出：
  - final metrics
  - per-family metrics
  - runtime counters
  - candidate funnel
  - workload summaries
  - metrics timeseries
- 支持 held-out workload 离线评估，用于比较 measured workload 和未优化查询上的 true-query error。

## 隐私边界

在 `privacy.mode=dp` 时，QDTE 优化只能使用 noisy/projected measurements 及其 variances：

```text
residual[q] = target_projected[q] - answer_syn[q]
measured_loss = 0.5 * sum_q residual[q]^2 * inv_variance[q]
delta[q] = phi_q(x_new) - phi_q(x_old)
edit advantage =
  delta @ (residual * inv_variance)
  - 0.5 * ((delta * delta) @ inv_variance)
  - lambda_cost * edit_cost
```

当前主线配置使用 `qdte.objective_weighting: unweighted`，即生成阶段设置优化用
`inv_variance[q] = 1`，直接拟合一致性投影后的 feasible target。measurement
variances 仍会保留在输出中用于审计和加权消融；旧的 inverse-variance objective
可通过 `qdte.objective_weighting: variance` 保留为 ablation。

exact true answers 只能用于离线 evaluation metrics。它们不会用于 active query selection、candidate generation、scoring、transport、stopping 或 hyperparameter selection，也不会写入 `measurements.json`。

held-out workload 也只用于离线评估：它不会进入 measurement 或 optimization loop。

当前支持两类 transport：`microbatch_greedy`/`sequential_greedy` prefix transport，以及 `atom_flow`。`atom_flow` 在候选 old-row/new-row encoded atom 图上选择一批 row edits，仍只使用 noisy/projected measurement residual。默认 `atom_flow_update_mode: batch` 会把 delta/score 和 prefix objective 放到 JAX 路径上批量计算，并在 engine 中一次性更新 residual；`atom_flow_update_mode: exact` 保留逐个 flow unit 精确 marginal 更新，适合小池对照。

## 主要目录

```text
configs/                 实验配置
scripts/                 命令行入口和环境检查脚本
qdte/
  config.py              YAML 配置读取和 dotted-key override
  dataio.py              JSON/NPY 输出辅助函数
  preprocess.py          CSV 读取、编码和解码
  schema.py              表 schema dataclass
  privacy/               zCDP accountant 和 Gaussian mechanism
  queries/               query catalogue、workload 构建、JAX query evaluation
  measurement/           DP/oracle measurement 和 projection
  evolution/             QDTE 初始化、调度、候选、评分、transport 和主 engine
  eval/                  metrics 和 runtime 统计
tests/                   单元测试和 smoke 测试
docs/                    handoff、任务记录和辅助文档
```

## 运行入口

常用入口是：

```bash
python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

当前环境中建议使用项目 conda 环境运行测试：

```bash
/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q
```

候选生成消融入口是：

```bash
/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_ablation.py \
  --config configs/smoke.yaml \
  --variant random_mutation \
  --qdte.max_iters 1000 \
  --qdte.stop_patience 1000
```

Population-level 可选入口是：

```bash
/home/qianqiu/.anaconda3/bin/conda run -n qdte python scripts/run_population.py \
  --config configs/smoke.yaml \
  --run.output_dir outputs/smoke_population \
  --population.size 4 \
  --population.elite_count 1 \
  --population.generations 10 \
  --population.inner_iters 200 \
  --population.parallel.enabled true \
  --population.parallel.gpu_devices 0,1 \
  --population.parallel.workers_per_gpu 1 \
  --population.crossover.enabled true \
  --population.crossover.mode context_aware \
  --population.crossover.children 2
```

该入口先运行一次 DP measurement/projection，再让多个 QDTE 个体复用同一个
`measurements.json`，最后按 measured objective 选择 elite。
`population.generations=1` 保留一轮 restart/elite wrapper 行为；
`population.generations>1` 会进入多代循环，每一代从上一代 elite clone、
crossover child 和 restart 个体中重新运行内层 QDTE。
`population.inner_iters` 是每个个体每一代的内层 QDTE 最大步数；若
`qdte.stop_patience` 触发，个体会提前停止。当前版本支持两种 crossover：

- `random_row`：从两个父 synthetic tables 随机抽取一部分 rows 生成 child；
- `context_aware`：把 donor parent 的 rows 当作接收 parent 的候选
  row-replacement edits，用接收 parent 的 residual 和 QDTE edit advantage
  重新评分，只接受正收益 replacements。

两种模式都会从生成的 child 继续运行 QDTE。candidate pool 不在不同 dataset
之间共享打分结果；共享的是 donor rows，advantage 始终按接收 dataset 的
residual 重新计算。

`population.parallel.enabled=true` 会启动持久 GPU worker pool。每个 worker
通过 `CUDA_VISIBLE_DEVICES` 绑定到 `population.parallel.gpu_devices` 中的
一个 GPU slot，并在多代循环中持续复用同一个 Python/JAX 进程，避免每个
dataset 反复初始化 JAX。当前推荐 `workers_per_gpu=1`；只有在确认单个 QDTE
个体显存很低且 `runtime.xla_preallocate=false` 时，再尝试同一卡多个 worker。

当前可选 candidate-generation variants 包括 `random_mutation`、`single_query`、`masked_single_query`、`paired_query`、`masked_paired_query`、`masked_exit_query`、`directed_exit_only`、`masked_exit_only` 和 `random_source_directed_exit`，其中多数也有 `_full` 或 `blind_` 消融形式。2026-06-10 的统一 1000-step smoke 消融显示：`random_mutation` 长跑 measured loss 最好；`masked_single_query` 略好于重跑的 `single_query` measured loss，但仍没有超过 random；full-budget paired 系列和 exit-only 系列更容易停滞或退化。

## 关键配置

基础 QDTE 配置在 `configs/adult_qdte.yaml`。高吞吐 GPU 配置在 `configs/adult_qdte_gpu_highpower.yaml`。

`configs/adult_qdte.yaml` 是质量优先配置，当前使用：

- `objective_weighting: unweighted`
- `transport_mode: atom_flow`
- `atom_flow_update_mode: batch`
- `atom_flow_pool_multiplier: 16`
- `atom_flow_max_pool: 0`
- `transport_prefix_strategy: best_advantage`
- `allow_below_noise_fallback: false`

两个 Adult 配置都默认开启 measurement consistency projection：

- `projection.prefix_monotonicity: true`
- `projection.consistency.enabled: true`
- `projection.consistency.method: local_marginal_ipf`

该投影使用已知行数作为硬约束：每个局部 marginal table 都满足非负且总和为 `N`，并通过重叠 scope 的 shared marginals 对齐 oneway/twoway/prefix/range/mixed/kway/kway_prefix/kway_range/kway_mixed/orthogonal_kway_mixed/halfspace 等查询之间的一致性。超出 `max_scope_cells` 的 scope 会 fail-fast，不会静默跳过。

`configs/adult_qdte_gpu_highpower.yaml` 是吞吐优先配置，当前显式使用：

- `objective_weighting: unweighted`
- `score_backend: sparse_delta_gpu`
- `candidate_backend: jax_repair`
- `transport_mode: atom_flow`
- `atom_flow_update_mode: batch`
- `atom_flow_pool_multiplier: 16`
- `atom_flow_max_pool: 0`
- `transport_delta_backend: jax_prefix`
- `total_candidates_per_iter: 1572864`
- `gpu_batches_per_iter: 1`
- `gpu_sparse_query_block_size: 64`
- `gpu_sparse_changed_attr_capacity: 4`
- `gpu_return_top_k: 8192`
- `accepted_per_iter: 1024`
- `allow_below_noise_fallback: true`

高吞吐配置当前默认使用 sparse-delta GPU scoring 和 batch atom-flow。`sparse_delta_gpu` 会把 attribute-to-query affected index、multi-word query scope bitsets 和 sparse query blocks 放进 fused JAX `pmap` scorer，避免每个 candidate 扫完整 query catalogue。50-iteration Adult 探针中，scoring time 从 dense GPU baseline 的约 `30.15s` 降到约 `14.26s`，总 generation time 从约 `36.65s` 降到约 `19.91s`。

需要回退到 dense GPU scoring 对照时，可覆盖：

```bash
--qdte.score_backend dense_gpu --qdte.gpu_score_query_block_size 1024
```

需要回退到 exact atom-flow 对照时，可覆盖：

```bash
--qdte.atom_flow_update_mode exact --qdte.atom_flow_max_pool 1024
```

held-out evaluation 可通过 `evaluation` 打开：

```yaml
evaluation:
  compute_true_query_error: true
  compute_heldout_query_error: true
  heldout_exclude_measured_queries: true
  heldout_workload:
    include_oneway: false
    include_2way_cat: true
    include_prefix: true
    include_range: true
    include_mixed: true
    include_kway: false
    include_kway_prefix: false
    include_kway_range: false
    include_kway_mixed: false
    include_orthogonal_kway_mixed: false
    include_halfspace: false
    max_queries: 10000
    max_terms: 4
    max_2way_cells: 10000
    range_intervals_per_num_attr: 128
    mixed_queries_per_pair: 128
    kway_orders: [3]
    kway_prefix_orders: [3]
    kway_range_orders: [3]
    kway_mixed_orders: [3]
    orthogonal_kway_mixed_orders: [2]
    kway_queries_per_order: 128
    kway_prefix_queries_per_order: 128
    kway_range_queries_per_order: 128
    kway_mixed_queries_per_order: 128
    orthogonal_kway_mixed_scopes_per_order: 16
    orthogonal_kway_mixed_range_bins: 4
    orthogonal_kway_mixed_max_cells_per_group: 4096
    exact_group_sensitivity_max_cells: 200000
    random_seed: 10000
```

## 主要输出

每次运行会写入配置里的 `run.output_dir`。常见输出包括：

- `config_resolved.yaml`
- `schema.json`
- `queries.json`
- `measurements.json`
- `synthetic_encoded.npy`
- `synthetic_decoded.csv`
- `metrics_final.json`
- `metrics_by_family.json`
- `metrics_timeseries.csv`
- `runtime.json`
- `workload_summary.json`
- `logs.txt`

开启 held-out evaluation 后还会输出：

- `queries_holdout.json`
- `workload_summary_holdout.json`
- `metrics_holdout.json`
- `metrics_by_family_holdout.json`

## 当前测试状态

最近一次验证命令：

```bash
/home/qianqiu/.anaconda3/bin/conda run -n qdte pytest -q
```

最近一次完整测试结果：

```text
76 passed in 6.92s
```

近期新增覆盖包括 directed repair enter/exit contract、paired/masked-paired/masked-exit compiler edge cases，以及 `sparse_delta_gpu` 跨 31 encoded attributes 边界的 multi-word bitset regression。

最近一次 highpower 代码优化基准：

```text
config: configs/adult_qdte_gpu_highpower.yaml
output: outputs/bench_gpu_sparsedelta_gpu_50
max_iters: 50
total_candidates_per_iter: 1572864
gpu_batches_per_iter: 1
score_backend: sparse_delta_gpu
gpu_sparse_query_block_size: 64
gpu_return_top_k: 8192
accepted_per_iter: 1024
result: 78,643,200 candidates scored
candidate_scoring_throughput_per_second: about 5.52M
candidates_scored_per_second: about 3.95M
accepted edits: 36,617
final measured loss: 127,132
dense GPU 50-iter baseline scoring time: about 30.15s
sparse-delta GPU 50-iter scoring time: about 14.26s
```

对比过的高候选规模：

```text
524,288 candidates/iter: ran, about 1.34M scoring candidates/s
655,360 candidates/iter: ran, about 1.38M scoring candidates/s
786,432 candidates/iter: ran, about 1.50M scoring candidates/s
1,048,576 candidates/iter + 2048 query blocks: ran, about 1.82M scoring candidates/s
1,572,864 candidates/iter + 1024 query blocks: ran, about 2.68M scoring candidates/s
2,097,152 candidates/iter + 1024 query blocks: ran, but dropped to about 2.26M scoring candidates/s
```

当前普通 shell 中 `pytest` 和 `python` 不在默认 PATH，直接运行 `pytest -q` 会失败；请优先使用上面的 conda 环境命令。

## 当前限制

- `kway`、`kway_prefix`、`kway_range`、`kway_mixed`、`orthogonal_kway_mixed` 默认关闭。启用后需要在 `privacy.measurement_allocation` 中给对应 family 分配预算，否则 DP measurement 会 fail-fast。
- `orthogonal_kway_mixed` 用完整 equality/range Cartesian partition 构造互斥 mixed queries；range term 来自等分的 disjoint intervals，因此每个 group 的 sensitivity 为 `1`。当前 halfspace query 仍是单阈值 `<=` 表示，不能直接表达互斥 linear slabs。
- halfspace 的 GPU fused single-query candidate repair 已支持 `jax_repair` / `gpu_repair`。复杂 CPU-only compilers 仍需要 `qdte.candidate_backend: cpu_repair`。
- `sparse_delta_gpu` 已使用 multi-word `uint32` query-scope bitsets，去掉了原先 31 个 encoded attributes 的单 word 限制；仍需保证 `gpu_sparse_changed_attr_capacity` 覆盖候选可能修改的属性数。
- 没有 adaptive query selection。
- 没有 public schema loader。
- 没有 plausibility materializer。
- 没有 baseline 系统。
- highpower GPU 配置默认仍偏吞吐，但当前使用 batch atom-flow transport。
- held-out workload 如果配置过窄，排除 measured duplicates 后可能剩余 0 个查询；实际实验应选择足够宽的 held-out workload。
