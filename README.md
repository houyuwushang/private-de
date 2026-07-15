# SAGE-QDTE

本仓库实现面向表格数据的 SAGE-QDTE 差分隐私合成系统。论文的科学主线是
Directed Evolution / QDTE：QDTE 把已发布的异构查询 residual 转换成
variance-aware 的精确 row-edit objective decrease。SAGE 是完整系统外壳，负责
公开 workload、SAGE-Select、DP measurement ledger、projection 和 QDTE
generation 的组织。

当前 paper-facing 方法明确区分：

- `SAGE-QDTE-Static`：静态完整 workload、P1 post-processing 和
  `QDTE-Standard`。
- `SAGE-QDTE-Adaptive`：只使用 released transcript 的 SAGE-Select、冻结的
  measurement ledger、P1 和 `QDTE-Standard`；当前作为完整选择变体报告，
  不替换 `SAGE-QDTE-Static` 默认配置。
- `QDTE-Structured-v2`、`QDTE-PA-Diag16`、Query-LSQ 和 RTP-local：固定目标
  生成器或 transfer/projection diagnostics，不替换默认完整方法。

当前 E1 冻结矩阵已经完成 340/340：四数据集、五个 epsilon、Static/Adaptive
SAGE-QDTE，以及强配置 AIM、MST 和官方 1M/full-N Private-GSD。E2--E6 分别
隔离 selector、projection、orthogonal workload、generator 和 component
composition。完整 E1 显示方法优势随 privacy regime 和 metric 变化：AIM 在最低
epsilon 的平均指标上很强，SAGE-QDTE 在中高 epsilon 的 RMSE/MaxError 更有优势，
Private-GSD 在部分 MaxTVD 单元领先。因此论文不主张逐数据集逐指标全面支配。
旧的 rho=1 QDTE package 保留为历史诊断，不是当前 SAGE-QDTE 主张来源。

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
- 支持 external row-level evaluator 和 paper table/figure packaging 脚本。

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

当前 paper-facing strong 配置使用 `qdte.objective_weighting: variance`。也就是说，QDTE 使用 measurement variances 构造 `inv_variance[q]`，低噪声 measurement 在 edit advantage 中权重更高。`qdte.objective_weighting: unweighted` 保留为消融或调试配置。

默认 Static 和 transcript-only Adaptive 路径不会把 exact true answers 用于
active query selection。实验性的 private-EM 路径只允许在具有显式全局敏感度
证明、指数机制采样和 selection privacy ledger 的受控 selector 内访问 exact
private block answers。无论使用哪条路径，exact truth 都不会进入 QDTE candidate
generation、scoring、transport、stopping 或 hyperparameter selection，也不会写入
`measurements.json`。

研究 runner 可能保存 `full_true_*` 和 `private_*` 离线审计字段。这些字段不是
可公开的 DP output；对外 release 只能包含声明的 noisy transcript、synthetic data
和 public metadata。完整边界见 `docs/DP_BOUNDARY.md`，外部代码审查入口见
`docs/CODE_REVIEW_GUIDE.md`。

可部署入口使用 `privacy.dp_release_mode: true`。该模式要求显式公开
`schema.json`（包括 category codebook、数值 bin edges 和 missing-value 规则）、
声明 `privacy.public_row_count: true` 并给出正整数
`privacy.public_n_rows`、固定 `privacy.adjacency: add_remove`，并禁止进程内
true-data evaluation；运行时私有输入行数必须与声明值一致，缺少任一条件都会
直接报错。声明式配置见
`configs/variants/dp_release_profile_overlay.yaml`，`scripts/run_sage_external.py`
会自动执行同一边界。隐私报告中的 `epsilon_delta` 由实际 `rho_spent` 计算，
`rho_total` 仅保留为声明上限。`measurements.json:privacy_ledger` 逐组记录
Gaussian vector mechanism、add/remove adjacency、L2 sensitivity、noise scale 和
实际 rho charge。

held-out workload 也只用于离线评估：它不会进入 measurement 或 optimization loop。

external evaluator 的 `full_true_*` 指标同样只属于离线评估。Evaluator 会在 `real_encoded.npy` 和 `synthetic_encoded.npy` 上精确回答同一个 public workload，按各自行数归一化，然后计算 `full_true_mae`、`full_true_rmse` 和 `full_true_max_error`。对于 partition/vector blocks，TVD 定义为 `0.5 * sum(abs(q(D_syn) / |D_syn| - q(D_real) / |D_real|))`；`full_true_avg_tvd` 和 `full_true_max_tvd` 分别是这些 block TVD 的均值和最大值。

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
  eval/                  metrics、runtime 统计和 external evaluator
tests/                   单元测试和 smoke 测试
docs/                    public release manifest；内部 paper/handoff 文档不建议公开
```

## 运行入口

最小 smoke 入口是：

```bash
conda run -n qdte python scripts/smoke_qdte.py --mode dp --rows 120 --max-iters 2
```

`scripts/smoke_qdte.py` 会自动生成 toy CSV，并在 `privacy.mode=dp`
下跑一个短 QDTE 端到端 smoke。若要使用自己的输入文件，可直接调用
`scripts/run_qdte.py` 和 YAML 配置。

paper-facing strong configs 是：

```text
configs/adult_sage_strong.yaml
configs/acs_sage_strong.yaml
configs/br2000_sage_strong.yaml
configs/nltcs_sage_strong.yaml
```

这些配置里的 `run.input_csv` 在本地实验环境中指向 canonical external input。公开复现时需要先按 release 文档准备同名输入，或者覆盖 `run.input_csv` 和 `run.output_dir`：

```bash
python scripts/run_qdte.py \
  --config configs/adult_sage_strong.yaml \
  --run.input_csv /path/to/adult_sage_strong/raw.csv \
  --run.output_dir outputs/adult_sage_strong_qdte
```

当前环境中建议使用项目 conda 环境运行测试：

```bash
conda run -n qdte pytest -q
```

部署型 DP 运行可以把私有测量和公开生成拆成两个进程。第一个进程是唯一
读取私有 CSV 的进程，并输出带哈希和实际隐私账本的公开 transcript；第二个
进程只读取该 transcript，配置校验会禁止 `run.input_csv` 和
`init.encoded_npy`：

```bash
conda run -n qdte python scripts/measure_qdte_transcript.py \
  --config /path/to/release_config.yaml \
  --output-dir outputs/public_transcript

conda run -n qdte python scripts/generate_qdte_from_transcript.py \
  --config /path/to/release_config.yaml \
  --transcript outputs/public_transcript \
  --output-dir outputs/public_generation
```

公开 transcript 只包含 `schema.json`、`queries.json`、
`measurements.json` 和 `transcript_manifest.json`。生成入口会先验证三份
payload 的 SHA-256、add/remove 邻接关系、公开行数和 actual-spend zCDP
账本，再进入 QDTE。

候选生成消融入口是：

```bash
conda run -n qdte python scripts/run_ablation.py \
  --config configs/smoke.yaml \
  --variant random_mutation \
  --qdte.max_iters 1000 \
  --qdte.stop_patience 1000
```

Population-level 可选入口是：

```bash
conda run -n qdte python scripts/run_population.py \
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

基础强实验配置是四个 `*_sage_strong.yaml` 文件。它们共同使用：

- `privacy.mode: dp`
- `privacy.measurement_mode: static_all`
- `qdte.objective_weighting: variance`
- `qdte.score_backend: dense_gpu`
- `qdte.transport_mode: atom_flow`
- `qdte.atom_flow_update_mode: batch`
- `qdte.total_candidates_per_iter: 4096`
- `qdte.accepted_per_iter: 64`
- `qdte.max_iters: 5000`

这些配置启用 partition projection、non-partition clipping，并按数据集启用 prefix monotonicity。当前主表强配置中，full local consistency projection 默认关闭：

```text
projection.consistency.enabled: false
```

`configs/adult_qdte_gpu_highpower.yaml` 是吞吐优先的旧 GPU 配置，当前显式使用：

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

## Paper-facing QDTE profiles

The current paper-facing method family is frozen as:

- `QDTE-Standard`: the end-to-end DP default.
- `QDTE-Structured`: the stronger controlled-generator profile using exact
  aggregate-delta two-row transport; it is not the DP default.
- `QDTE-FissionRefit`: a released-only, two-pass MAE/RMSE alignment variant.

Encoded attribute cardinalities are treated as public and known. The CSV loader
may infer them as a convenience when no separate schema file is supplied; this
is not a private schema-estimation claim.

Generate and verify the current external evidence package with:

```bash
conda run -n qdte python scripts/package_qdte_paper_results.py --force
conda run -n qdte python scripts/verify_qdte_paper_package.py
conda run -n qdte python scripts/archive_qdte_paper_package.py
conda run -n qdte python scripts/verify_paper_package_tarball.py
```

The controlled same-target GSD comparison uses
`scripts/materialize_gsd_measurement.py`,
`scripts/run_official_gsd_on_qdte_workload.py`, and the frozen manifests under
`configs/variants/qdte_gsd_*_seed0_manifest.yaml`. FissionRefit uses
`scripts/run_qdte_fission_refit_external.py` with the Standard-v2,
search-aware, and fission-refit-v2 overlays. Exact true answers remain offline
evaluation artifacts and never select a DP candidate, checkpoint, or config.

## 当前测试状态

推荐验证命令：

```bash
conda run -n qdte pytest -q
```

GPU/JAX 环境检查：

```bash
conda run -n qdte python scripts/check_env.py
```

Paper-facing GPU provenance audit：

```bash
python3 scripts/audit_gpu_provenance.py
```

该 audit 检查当前 strict same-protocol 结果中 QDTE-Standard、Private-GSD 和 RAP
的 GPU metadata，并把 AIM/MST 明确标记为 CPU-native 行。

Original-protocol reproduced baseline audit：

```bash
python3 scripts/audit_original_protocol_baselines.py
```

该 audit 检查 RAP++ official ACS/Folktables grid 和 PrivMRF official TVD
epsilon grid 的原协议复现证据；这些结果和 strict same-protocol 主表分开报告。

近期覆盖包括 measurement/projection、query workload、external evaluator、edit advantage、GPU candidate path、transport、scheduler 和 engine smoke tests。具体通过数量以当前运行输出为准。

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
- 主表强配置使用 public static measurement schedule；旧 adaptive selector runner 不属于当前 QDTE 论文主张。
- external baseline repositories 不 vendored 到本仓库；相关 wrapper/collector 依赖外部 baseline workspace 和独立 conda 环境。
- baseline 证据分层以 `docs/EXTERNAL_BASELINES.md` 和 `docs/PUBLIC_RELEASE_MANIFEST_20260706.md` 为准；official 原协议复现结果作为 paper artifact 保存，不进入源码仓库。
- public schema loader 仍不完整；公开复现建议从 CSV 和 config 重新生成 schema/workload。
- plausibility materializer 仍未实现。
- highpower GPU 配置默认仍偏吞吐，但当前使用 batch atom-flow transport。
- held-out workload 如果配置过窄，排除 measured duplicates 后可能剩余 0 个查询；实际实验应选择足够宽的 held-out workload。

## Public release manifest

公开仓库清理规则见：

```text
docs/PUBLIC_RELEASE_MANIFEST_20260706.md
```

内部 handoff、专家讨论、论文草稿和 scratch outputs 不应进入 public release branch。
