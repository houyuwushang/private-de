# External Baselines

This document describes how QDTE external baseline comparisons are organized. Baseline repositories are not vendored into this source repository.

## 1. Shared Contract

Every method in the external comparison should:

1. read the same canonical input package;
2. use the same privacy budget and delta;
3. produce a row-level encoded synthetic table;
4. write run metadata;
5. be evaluated by the shared offline evaluator.

The required synthetic output is:

```text
synthetic_encoded.npy
```

The shared evaluator writes:

```text
evaluation.json
```

## 2. External Baseline Workspace

External baseline repositories and generated artifacts should live outside this
source repository. The scripts default to `external_workspace/`; set
`SAGE_BASELINE_ROOT` to point at a different external workspace:

```text
$SAGE_BASELINE_ROOT/
```

That workspace is intentionally outside this repository. A public release should
document how to obtain or configure each upstream baseline rather than copying
external code here.

## 3. Planner Methods

`scripts/plan_external_experiments.py` recognizes these method keys:

```text
sage
private_gsd_gpu
private_gsd_gpu_1m_fulln_audit
private_gsd
private_gsd_stronger
private_pgm_aim
private_pgm_mst
dpmm_aim
dpmm_mst
dpmm_privbayes
datasynth_privbayes
rap
gem
rappp_marginal
privmrf
privmrf_gpu
privsyn_unofficial
```

The main paper table should use the strict, best-audited same-protocol rows:

```text
QDTE-Standard
RAP softmax
Private-GSD GPU 1M/full-N
Private-PGM AIM
Private-PGM MST
```

GEM, PrivBayes, PrivSyn, and QDTE-canonical PrivMRF wrappers are diagnostic or
audit baselines unless their wrappers satisfy the same evidence bar. RAP++
official and PrivMRF official are tracked as upstream original-protocol
reproductions and must be reported separately from the strict same-protocol
main table.
The concrete admission criteria and current per-method decisions are recorded in:

```text
docs/EXTERNAL_BASELINE_ADMISSION_MATRIX_20260706.md
```

## 4. GPU Provenance Gate

Paper-facing GPU-capable methods must carry machine-checkable GPU evidence in
their run artifacts. Run this audit after full-grid or package refreshes:

```bash
python3 scripts/audit_gpu_provenance.py
```

The audit checks the current seed0-4 paper-facing runs:

```text
QDTE-Standard
Private-GSD GPU 1M/full-N
RAP softmax
Private-PGM AIM
Private-PGM MST
```

QDTE-Standard, Private-GSD, and RAP are required to expose GPU evidence through
`runtime.json`, `metrics_final.json`, or `run_metadata.json`. Private-PGM AIM
and MST are marked `cpu_native` because the upstream MBI/private-pgm path is
not GPU-native in this environment. This distinction is important: a CPU-native
canonical baseline is acceptable, but a GPU-capable baseline silently falling
back to CPU is not.

The external planner also performs an execution-time preflight for selected
GPU-required environments. With the default `--gpu-preflight` setting,
`scripts/plan_external_experiments.py` prints one probe per selected
environment in dry-run mode and runs the same probe before experiments under
`--execute`. The current probes are JAX GPU visibility for QDTE and
Private-GSD GPU, Torch CUDA visibility for RAP/GEM, and CuPy device visibility
for `privmrf_gpu`. Use `--no-gpu-preflight` only for deliberate CPU debugging;
paper-facing GPU-capable rows still need to pass `scripts/audit_gpu_provenance.py`.

## 5. Expected Environments

The planner currently maps methods to conda environments:

```text
sage                 qdte
private_gsd_gpu      gsd
private_gsd_gpu_1m_fulln_audit  gsd
private_gsd          baseline_gsd
private_pgm_aim      baseline_mbi
private_pgm_mst      baseline_mbi
dpmm_*               baseline_dpmm
datasynth_privbayes  baseline_datasynth
rap                  tddpm
gem                  tddpm
rappp_marginal       qdte
privmrf*             baseline_privmrf
privsyn_unofficial   tddpm
```

These environment names are local conventions. Public reproduction should either recreate them or edit the planner.

## 6. Example Main-Table Run

Plan commands:

```bash
conda run -n qdte python scripts/plan_external_experiments.py \
  --phase full \
  --methods sage,rap,private_gsd_gpu_1m_fulln_audit,private_pgm_aim,private_pgm_mst \
  --datasets adult_sage_strong,acs_sage_strong,br2000_sage_strong,nltcs_sage_strong \
  --rhos 1.0 \
  --seeds 0,1,2,3,4
```

Execute and evaluate:

```bash
conda run -n qdte python scripts/plan_external_experiments.py \
  --phase full \
  --methods sage,rap,private_gsd_gpu_1m_fulln_audit,private_pgm_aim,private_pgm_mst \
  --datasets adult_sage_strong,acs_sage_strong,br2000_sage_strong,nltcs_sage_strong \
  --rhos 1.0 \
  --seeds 0,1,2,3,4 \
  --execute \
  --evaluate
```

The default run plan includes GPU preflight commands before the experiment
commands for QDTE-Standard, Private-GSD GPU, and RAP. These probes are intentionally
outside the privacy accounting because they inspect only the local execution
environment.

## 7. Strong Private-GSD GPU Notes

The paper-facing GSD row should use the GPU-backed wrapper when available:

```text
method key = private_gsd_gpu_1m_fulln_audit
conda env = gsd
```

Current paper-facing settings are documented in the result package and table notes. Important parameters include:

```text
N_prime
tree_query_depth
genetic_operators
num_generations
stop_early_min_generation
early_stop_threshold
```

The current paper-facing seed0-4 result package promotes the high-power
GPU-backed `gsd` configuration:

```text
N_prime = n
tree_query_depth = 2
genetic_operators = mutate,swap,cross
num_generations = 1000000
stop_early_min_generation = 1000000
early_stop_threshold = 0.01, not active before the one-million-generation budget
```

The older 200k-generation, `N_prime=2048`, early-stop configuration is retained
only as a configuration/runtime sensitivity row. It should not be described as
the current main Private-GSD comparison.

The high-power GSD row changes the claim: QDTE-Standard wins MAE, RMSE, and MaxErr on
all four datasets and wins all five metrics on Adult and NLTCS, while
high-power GSD is lower on AvgTVD and MaxTVD for ACS and BR2000.

## 8. AIM and MST Notes

Private-PGM AIM and MST use:

```text
method keys = private_pgm_aim, private_pgm_mst
conda env = baseline_mbi
```

AIM can be memory-sensitive on some canonical workloads. If fallback settings are used, report:

```text
rounds
max_iters
max_model_size
failure log or reason for fallback
```

The frozen strong AIM workloads are run serially and require the Linux host
setting below:

```text
vm.max_map_count >= 262144
```

On the 32-core evaluation host, ACS and NLTCS exceeded the Linux default of
65,530 executable mappings during JAX/XLA compilation even though more than
100 GiB of physical memory remained available. The resulting LLVM section
allocation error is a host resource ceiling, not an AIM model-size failure.
The WP4 runner checks this setting before launching incomplete AIM jobs and
records both the setting and inherited CPU affinity in every execution record.
For a temporary setting that lasts until reboot:

```bash
sudo sysctl -w vm.max_map_count=262144
```

Do not reduce `rounds`, `max_iters`, or `max_model_size` to bypass this gate.
If the precondition cannot be met, preserve the failure as missing baseline
evidence rather than silently substituting a weaker AIM configuration.

## 9. Additional Baseline Wrappers And Diagnostics

### RAP

RAP softmax is now primary row-level evidence when run through the shared
evaluator. It should still be described distinctly because its optimization and
row-generation path differ from QDTE-Standard, Private-GSD, AIM, and MST.

Current paper-package RAP rows are seed0-4 GPU-backed `rap_softmax` runs in the
`tddpm` environment with `torch_cuda_available=True`,
`torch_device=cuda:0`, `T=30`, `K=30`, and `max_iters=1000`.

### RAP++

The current canonical wrapper key is:

```text
rappp_marginal
```

This uses the RAP++ projection implementation but only marginal statistics on
the encoded canonical input package. It is a path check, not official full
RAP++ evidence.

Paper-facing RAP++ admission should first use the original repository's own
interface and defaults, or the paper-aligned settings reported by the authors.
If that route reaches the originally advertised level, it should be treated as
a comparable external baseline. QDTE-specific shared-budget and shared-workload
runs are still useful, but they belong in controlled mechanism comparisons or
ablation sections rather than serving as a way to weaken an external method.

The upstream implementation is JAX-based. Its published `requirements.txt`
pins `jax==0.2.7` and `jaxlib==0.1.57`; the local `baseline_rap` environment
matches that stack but only sees `cpu:0` on this RTX 4090 machine. The current
GPU-backed official-code runs therefore use the `qdte` environment with modern
JAX and record JAX/JAXLIB/device metadata in `run_metadata.json`.

The official RAP++ repository path has also been smoke-tested and extended on
its natural ACS/Folktables interface:

```text
$SAGE_BASELINE_ROOT/relaxed-adaptive-projection
```

The successful smoke used `qdte` with JAX GPU, `folktables==0.0.12`, CA income,
and the upstream `RAP(Marginal&Halfspace)` configuration with a small random
projection count:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n qdte python main.py \
  --states CA \
  --targets income \
  --algorithm RAP++ \
  --seed 0 \
  --epsilon 1.0 \
  --k 2 \
  --num_random_projections 64 \
  --top_q 1 \
  --dp_select_epochs 1
```

It produced:

```text
results/sync_data/RAP(Marginal&Halfspace)/acs_CA_income/1.00/(1, 1)/0/synthetic.csv
```

This official ACS output is not directly comparable to the current
`acs_sage_strong` main-table row: RAP++ uses raw Folktables ACS features with
continuous columns and target-conditioned halfspaces, while `acs_sage_strong`
uses a 23-column encoded canonical package and a heterogeneous QDTE workload
without halfspace queries. To promote official RAP++ into a paper-facing
comparison, create a separate same-protocol ACS/Folktables experiment and run
QDTE-Standard and RAP++ under that shared input, workload, budget, and evaluator.

The first same-protocol smoke package is now available:

```text
input package = $SAGE_BASELINE_ROOT/external_inputs/acs_ca_income_rappp_sage
summary = $SAGE_BASELINE_ROOT/external_results/rappp_official_acs_smoke_20260706.md
```

The current original-protocol ACS paper-grid reproduction is:

```text
states = NY, CA, TX, FL, PA
tasks = income, travel, coverage, employment, mobility
seeds = 0, 1, 2, 3, 4
num_random_projections = 200000
top_q = 5
dp_select_epochs = 50
result = $SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0to4_20260707.md
```

Seeds 1 through 4 have complete GPU metadata: all 100 runs report
`jax_device_platforms=gpu` and `jax_devices=cuda:0`. Seeds 1 and 3 used
`CUDA_VISIBLE_DEVICES=0`; seeds 2 and 4 used `CUDA_VISIBLE_DEVICES=1`.
Seed0 predates the metadata probe and remains useful for original-protocol
metrics but not for device-audit claims.

Helper scripts:

```text
scripts/export_rappp_acs_folktables.py
scripts/create_external_input_package.py
scripts/encode_external_synthetic_csv.py
configs/acs_ca_income_rappp_sage.yaml
```

This smoke confirms that a same-protocol QDTE-versus-RAP++ ACS bridge can run
end-to-end. It is not final same-protocol evidence: the bridge has only small
RAP++ projection-count probes and short QDTE iteration sweeps. The official
RAP++ evidence used for paper coverage is the upstream original-protocol
seed0-4 grid above, not this bridge smoke.

The official RAP++ wrapper now defaults to the upstream code defaults:

```text
epsilon = 1.0
k = 2
num_random_projections = 200000
top_q = 5
dp_select_epochs = 50
```

For reproduction, call it with `--upstream-epsilon 1.0`. For QDTE budget
alignment, call it with `--rho-total ...`; the wrapper records the conversion
in `run_metadata.json`.

The first upstream-default ACS CA income reproduction completed after a
chunked-statistics compatibility patch that reduces peak memory without
changing the RAP++ configuration. The run used `epsilon=1.0`,
`num_random_projections=200000`, `top_q=5`, and `dp_select_epochs=50`.

Current shared-evaluator metrics for that row:

```text
MAE    = 0.008015
RMSE   = 0.037529
AvgTVD = 0.300405
MaxErr = 0.894947
MaxTVD = 0.930722
```

Keep this row separate from the rho-aligned smoke rows: upstream `epsilon=1.0`
corresponds to `rho_total ~= 0.010291` under RAP++'s internal
`delta=1/n^2`.

Original-style RAP++ metrics for the same output are also available:

```text
paper metrics = $SAGE_BASELINE_ROOT/external_runs/rappp_official_acs/acs_ca_income_rappp_sage/upstream_eps1/seed0_code_default_chunk20k/paper_metrics/rappp_paper_metrics.json
```

Key CA income values:

```text
conditional 2-way marginal average error = 0.089595
random mixed-prefix average error        = 0.005008
real-train LR macro F1                   = 0.958499
RAP++ synthetic LR macro F1              = 0.935469
```

These metrics are closer to RAP++'s original paper protocol than the shared
QDTE evaluator. Use them when deciding whether RAP++ has been reproduced at
its claimed level.

Historical seed0-only archive, superseded by the seed0-4 grid above:

RAP++ was first run over the original-paper ACS grid for seed 0:

```text
states = NY, CA, TX, FL, PA
tasks  = income, travel, coverage, employment, mobility
```

The grid runner uses the upstream ACS/Folktables interface and RAP++ defaults,
with only a chunked-statistics implementation patch to fit the full
`200000`-projection run on the local RTX 4090:

```text
runner = scripts/run_rappp_official_paper_grid.py
epsilon = 1.0
k = 2
num_random_projections = 200000
top_q = 5
dp_select_epochs = 50
RAPPP_FULL_STATS_CHUNK_SIZE = 20000
completed = 25/25
```

Collected artifacts:

```text
$SAGE_BASELINE_ROOT/external_runs/rappp_official_paper_grid/
$SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0_20260706.csv
$SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0_20260706_by_target.csv
$SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0_20260706_by_state.csv
$SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0_20260706_overall.csv
$SAGE_BASELINE_ROOT/external_results/rappp_official_paper_grid_seed0_20260706.md
```

Seed-0 overall original-protocol results:

```text
conditional 2-way marginal average error = 0.122129
20K random mixed-prefix average error    = 0.005154
synthetic LR macro F1                    = 0.742445
macro-F1 gap to real train               = 0.021320
mean task runtime                        = 93.48s
```

By-target summary:

| Target | Marginal avg. error | Prefix avg. error | Synthetic macro F1 | Macro-F1 gap | Runtime |
|---|---:|---:|---:|---:|---:|
| coverage | 0.100305 | 0.007228 | 0.715389 | 0.006437 | 96.88s |
| employment | 0.081126 | 0.003925 | 0.968776 | 0.012616 | 107.56s |
| income | 0.114989 | 0.005461 | 0.915210 | 0.036435 | 71.95s |
| mobility | 0.202867 | 0.004079 | 0.525648 | 0.039195 | 100.97s |
| travel | 0.111358 | 0.005076 | 0.587203 | 0.011916 | 90.05s |

Historical admission note:

- The current paper-facing decision is the seed0-4 original-protocol reproduced
  row above.
- Do not force RAP++ into QDTE's internal hyperparameter choices for the main
  baseline comparison.
- Use QDTE-matched parameters only for controlled ablations and mechanism
  studies, where the purpose is to isolate query selection, measurement, and
  generation choices rather than to reproduce an external method at its best
  reported operating point.

### GEM

GEM uses an RDT categorical transformer whose one-hot positions follow first-seen
category order, while the iterative-DP `QueryManager` uses sorted integer
category ids. The wrapper therefore remaps public query coordinates before
training and writes `transformer_query_remap.json` for auditability. The remap
removes the earlier relaxed-to-row artifact. The remapped seed0 all-dataset
diagnostic remains much weaker than the primary baselines under the shared
strong evaluator, so GEM remains appendix/diagnostic evidence rather than a
main-table candidate.

### PrivBayes, PrivSyn, PrivMRF

These methods should appear in appendix/audit tables unless they are reproduced
under their original public protocols or fully calibrated under the strict
same-protocol evidence bar. Unofficial wrappers must be labeled as unofficial.

Current original-protocol status:

| Method | Current status | Paper-facing decision |
|---|---|---|
| DataSynthesizer PrivBayes | Official package/notebook entry exists; local `baseline_datasynth` import works; current result is one Adult strong seed-0 QDTE-canonical audit row. | Appendix/audit now. If upgraded, first reproduce the DataSynthesizer correlated-attribute notebook/public-data protocol rather than expanding QDTE-tuned parameters. |
| DPMM PrivBayes | Public library entry exists and imports in `baseline_dpmm`, but current Adult calibration is very weak. | Calibration/failure appendix unless a different implementation or research question is chosen. |
| PrivSyn | Local repository is an unofficial course-project implementation with notebook-first assumptions and wrapper-generated configuration patches. | Transparency appendix only; label `unofficial`. Do not promote without an official artifact/protocol. |
| PrivMRF | Official repository and `script.py` reproduction entry exist; the local official TVD grid now runs through `scripts/run_privmrf_official.py` for `nltcs`, `acs`, `adult`, and `br2000`. | Original-protocol reproduced evidence for TVD. Keep separate from QDTE's shared evaluator; use QDTE-matched PrivMRF rows only for ablations. |

Known current audit values:

```text
DataSynthesizer Adult seed0: MAE=0.00216874, RMSE=0.00699662, AvgTVD=0.0638604
DPMM PrivBayes Adult seed0: MAE=0.0280011, RMSE=0.0840949, AvgTVD=0.707552
PrivSyn unofficial Adult seed0: MAE=0.000976435, RMSE=0.00460734, AvgTVD=0.0472857
PrivMRF GPU Adult seed0: MAE=0.001678, AvgTVD=0.05511
PrivMRF GPU BR2000 seed0: MAE=0.003787, AvgTVD=0.10633
```

The local PrivMRF official-path environment has been repaired enough for a
minimal smoke: `scikit-learn` was installed into `baseline_privmrf`, and
`scripts/run_privmrf_official.py` completed an upstream API call on
`nltcs`, `epsilon=0.8`, `repeat=1`, `marginal_num=5`.

Smoke artifact:

```text
$SAGE_BASELINE_ROOT/external_runs/privmrf_official/official_smoke_nltcs_eps0p8_m5_20260706/
```

Smoke TVD values:

```text
3-way = 0.004732
4-way = 0.007935
5-way = 0.011639
```

This confirms the official API path is now runnable. It is not yet
original-paper-level evidence because the paper reproduction uses the full
dataset/epsilon grid and `marginal_num=300`.

The full upstream-parameter TVD grid has completed:

```text
runner = scripts/run_privmrf_official.py
collector = scripts/collect_privmrf_official.py
datasets = nltcs, acs, adult, br2000
epsilons = 0.1, 0.2, 0.4, 0.8, 1.6, 3.2
repeat = 1
marginal_num = 300
runtime = 140.63s + 2647.64s
```

Artifacts:

```text
$SAGE_BASELINE_ROOT/external_runs/privmrf_official/official_nltcs_tvd_epsgrid_m300_20260706/
$SAGE_BASELINE_ROOT/external_runs/privmrf_official/official_acs_adult_br2000_tvd_epsgrid_m300_20260706/
$SAGE_BASELINE_ROOT/external_results/privmrf_official_nltcs_tvd_epsgrid_m300_20260706.csv
$SAGE_BASELINE_ROOT/external_results/privmrf_official_nltcs_tvd_epsgrid_m300_20260706.md
$SAGE_BASELINE_ROOT/external_results/privmrf_official_acs_adult_br2000_tvd_epsgrid_m300_20260706.csv
$SAGE_BASELINE_ROOT/external_results/privmrf_official_acs_adult_br2000_tvd_epsgrid_m300_20260706.md
$SAGE_BASELINE_ROOT/external_results/privmrf_official_full_tvd_epsgrid_m300_20260706.csv
$SAGE_BASELINE_ROOT/external_results/privmrf_official_full_tvd_epsgrid_m300_20260706.md
```

| Dataset | Epsilon | 3-way TVD | 4-way TVD | 5-way TVD |
|---|---:|---:|---:|---:|
| nltcs | 0.1 | 0.018216 | 0.028788 | 0.041515 |
| nltcs | 0.2 | 0.015241 | 0.023274 | 0.032719 |
| nltcs | 0.4 | 0.007356 | 0.011572 | 0.017495 |
| nltcs | 0.8 | 0.004853 | 0.007960 | 0.011665 |
| nltcs | 1.6 | 0.003026 | 0.005209 | 0.008398 |
| nltcs | 3.2 | 0.002435 | 0.003874 | 0.006215 |
| acs | 0.1 | 0.013551 | 0.020761 | 0.027800 |
| acs | 0.2 | 0.009944 | 0.014521 | 0.020405 |
| acs | 0.4 | 0.005253 | 0.008200 | 0.011736 |
| acs | 0.8 | 0.004023 | 0.005780 | 0.008310 |
| acs | 1.6 | 0.002472 | 0.003759 | 0.005634 |
| acs | 3.2 | 0.001722 | 0.002675 | 0.004304 |
| adult | 0.1 | 0.127128 | 0.198920 | 0.277111 |
| adult | 0.2 | 0.108537 | 0.161308 | 0.240660 |
| adult | 0.4 | 0.087550 | 0.152520 | 0.206831 |
| adult | 0.8 | 0.044208 | 0.077721 | 0.122651 |
| adult | 1.6 | 0.039715 | 0.072034 | 0.116838 |
| adult | 3.2 | 0.034298 | 0.066165 | 0.108989 |
| br2000 | 0.1 | 0.094177 | 0.159474 | 0.218203 |
| br2000 | 0.2 | 0.076708 | 0.116337 | 0.177182 |
| br2000 | 0.4 | 0.051233 | 0.102841 | 0.144770 |
| br2000 | 0.8 | 0.027523 | 0.052411 | 0.082395 |
| br2000 | 1.6 | 0.020164 | 0.036381 | 0.065383 |
| br2000 | 3.2 | 0.016800 | 0.033197 | 0.058994 |

This is original-protocol reproduced evidence for PrivMRF's TVD experiment. It
should not be merged into the strict QDTE evaluator table because the metric,
budget convention, and workload are PrivMRF-native.

PrivMRF uses an upstream internal dataset-name switch for some algorithm branches. The planners now default to:

```text
--privmrf-data-name auto
```

This maps canonical datasets such as `adult_sage_strong` and `br2000_sage_strong` back to the upstream names `adult` and `br2000`, which is necessary for the original PrivMRF GPU path. Keep this field in run metadata when reporting PrivMRF official reproduced evidence or QDTE-matched PrivMRF audit rows.

## 9. Collecting Results

After runs finish:

```bash
conda run -n qdte python scripts/collect_external_results.py \
  --runs-root /path/to/external_runs \
  --output-csv /path/to/external_results/summary_all.csv \
  --output-md /path/to/external_results/summary_all.md
```

Plot:

```bash
conda run -n qdte python scripts/plot_external_results.py \
  --summary-csv /path/to/external_results/summary_all.csv \
  --out-dir /path/to/external_results/figures \
  --metrics full_true_mae,full_true_rmse,full_true_avg_tvd,full_true_max_error,full_true_max_tvd,runtime_seconds \
  --rho 1.0 \
  --formats pdf,png
```

Audit the original-protocol reproduced grids without launching experiments:

```bash
python3 scripts/audit_original_protocol_baselines.py
```

This verifies the RAP++ official ACS grid protocol and the PrivMRF official TVD
epsilon grid separately from the strict QDTE shared-evaluator table.

## 10. Fairness Notes

The strict table, original-protocol reproduced table, and appendix tiers are
intentional:

```text
strict same-protocol main table = strongest clean shared-evaluator rows
original-protocol reproduced table = upstream public-code/native-metric evidence
appendix/audit = single-seed, unofficial, weaker, or diagnostic rows
```

Do not silently omit credible baseline evidence. Do not merge incomparable
wrappers or native-protocol metrics into the strict main table without
explaining their status.
