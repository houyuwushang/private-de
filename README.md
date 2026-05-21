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
- 支持 DP measurement：
  - 使用 zCDP Gaussian mechanism 加噪。
  - 对 partition workload 可做 projection。
  - 对非 partition counts 可做 clipping。
- 支持 oracle mode，用于 debug 或上界实验，不应作为 DP 结果使用。
- 支持 QDTE edit loop：
  - active query selection
  - CPU repair candidate generation
  - dense JAX/GPU scoring
  - non-conflicting edit selection
  - batch transport acceptance
  - periodic full recompute drift check
- 支持 GPU-oriented candidate path：
  - `jax_repair` / `gpu_repair`
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

exact true answers 只能用于离线 evaluation metrics。它们不会用于 active query selection、candidate generation、scoring、transport、stopping 或 hyperparameter selection，也不会写入 `measurements.json`。

held-out workload 也只用于离线评估：它不会进入 measurement 或 optimization loop。

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

## 关键配置

基础 QDTE 配置在 `configs/adult_qdte.yaml`。高吞吐 GPU 配置在 `configs/adult_qdte_gpu_highpower.yaml`。

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
    include_halfspace: false
    max_queries: 10000
    max_terms: 4
    max_2way_cells: 10000
    range_intervals_per_num_attr: 128
    mixed_queries_per_pair: 128
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

结果：

```text
15 passed in 19.63s
```

当前普通 shell 中 `pytest` 和 `python` 不在默认 PATH，直接运行 `pytest -q` 会失败；请优先使用上面的 conda 环境命令。

## 当前限制

- halfspace workload 尚未实现。
- 没有 adaptive query selection。
- 没有 public schema loader。
- 没有 debt scheduler。
- 没有 plausibility materializer。
- 没有 baseline 系统。
- held-out workload 如果配置过窄，排除 measured duplicates 后可能剩余 0 个查询；实际实验应选择足够宽的 held-out workload。
