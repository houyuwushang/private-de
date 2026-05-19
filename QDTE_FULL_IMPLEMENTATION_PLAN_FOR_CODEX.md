# QDTE / Query-wise Directed Transport Evolution: Full Implementation Plan for Codex

> Goal: implement a real, runnable QDTE algorithm for DP synthetic tabular data generation. This is not a toy simulation. The implementation must read a real tabular dataset, build real statistical-query workloads, measure those queries on the real dataset under a DP mechanism, and then run QDTE generation against the noisy measured targets.
>
> Important terminology in this plan:
> - **Real measurement** means: evaluate queries on the real/private dataset and add DP noise according to the allocated privacy budget. This is the default paper-valid mode.
> - **Oracle measurement** means: evaluate queries on the real/private dataset without DP noise. This is only for debugging and upper-bound experiments. It is not DP and must never be described as the released algorithm.
> - **Simulated measurement** is not the target here. Do not create fake target vectors unless running unit tests.

---

## 0. High-level implementation target

Implement the following pipeline:

```text
Real tabular data D
  -> schema inference / preprocessing / discretization
  -> candidate workload construction
  -> DP measurement on real D
  -> optional consistency projection / target cleanup
  -> initial synthetic data D_hat
  -> QDTE generation:
       residual r = a_bar - Q(D_hat)
       active query scheduler
       query-specific candidate edit generation
       exact edit-impact scoring
       greedy or micro-batch transport
       incremental residual/state update
       periodic full recomputation
  -> synthetic data output
  -> utility/runtime diagnostics
```

The first complete version must be able to run end-to-end on datasets with roughly:

```text
N = 20,000 to 50,000 rows
D = 10 to 30 attributes
m = 1,000 to 20,000 measured queries
2 x RTX 4090 GPUs
```

The implementation should prioritize a correct, full run over a theoretically perfect implementation. Use dense GPU query scoring first, then add compiled-oracle scoring if needed.

---

## 1. Framework choice

Use **JAX-first**.

Reason:

1. Private-GSD is fast partly because JAX/XLA can compile batched objective evaluations.
2. QDTE is mostly batched Boolean query evaluation, batched candidate edit generation, and batched matrix/vector scoring. These are well suited to JAX.
3. Multi-GPU candidate scoring can be implemented by splitting candidate batches across devices with `jax.pmap` or `jax.device_put_sharded`.
4. Tabular data sizes are moderate, so dense `[batch_candidates, num_queries]` Boolean query-impact evaluation on GPU may already be fast enough.

Do not build neural-network machinery unless downstream ML evaluation needs it. PyTorch is optional only for comparison or downstream classifiers. The QDTE core should be JAX/NumPy/Pandas/Scikit-learn.

---

## 2. Environment setup

Create a conda environment, then install JAX through pip. Use CUDA 12 or CUDA 13 depending on the server driver.

### 2.1 Check GPU and driver

```bash
nvidia-smi
```

Expected: two RTX 4090 devices visible.

### 2.2 Create environment

```bash
conda create -n qdte python=3.11 -y
conda activate qdte
python -m pip install --upgrade pip setuptools wheel
```

### 2.3 Install JAX GPU build

Try CUDA 12 first if the current machine uses a normal CUDA 12 stack:

```bash
pip install -U "jax[cuda12]"
```

If the driver is new enough and CUDA 13 is preferred:

```bash
pip install -U "jax[cuda13]"
```

### 2.4 Install other dependencies

```bash
pip install -U numpy pandas scipy scikit-learn pyyaml tqdm rich matplotlib seaborn pytest pyarrow orjson
pip install -U typer loguru
```

`seaborn` is only for local experiment plots; the core code must not depend on it.

### 2.5 Verify JAX GPU visibility

Create `scripts/check_env.py`:

```python
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Local device count:", jax.local_device_count())

x = jnp.ones((4096, 4096), dtype=jnp.float32)
y = x @ x
print("Result:", float(y[0, 0]))
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/check_env.py
```

The output must show two GPU devices.

Optional memory setting during development:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For maximum performance during final runs, allow JAX to preallocate GPU memory unless it causes conflicts.

---

## 3. Repository layout

Implement the codebase with this structure:

```text
qdte/
  __init__.py
  config.py
  schema.py
  preprocess.py
  dataio.py

  queries/
    __init__.py
    types.py
    workload.py
    eval_jax.py
    repair.py

  privacy/
    __init__.py
    accountant.py
    gaussian.py
    exponential.py

  measurement/
    __init__.py
    measure.py
    projection.py

  evolution/
    __init__.py
    state.py
    scheduler.py
    candidates.py
    scoring.py
    transport.py
    engine.py

  eval/
    __init__.py
    metrics.py
    downstream.py
    runtime.py

scripts/
  check_env.py
  smoke_qdte.py
  run_qdte.py
  run_ablation.py
  run_private_gsd_baseline.py    # optional wrapper if existing baseline code is available

configs/
  adult_qdte.yaml
  acs_qdte.yaml
  smoke.yaml

tests/
  test_queries.py
  test_measurement.py
  test_edit_advantage.py
  test_repairs.py
  test_engine_smoke.py

outputs/
  .gitkeep
```

---

## 4. Data model and preprocessing

### 4.1 Internal representation

Use integer-coded arrays for the QDTE core:

```python
X_real: int32 array [N, d]
X_syn:  int32 array [N_syn, d]
```

Every column has a finite domain:

```python
num_categories[attr] = K_attr
valid values are 0, 1, ..., K_attr - 1
```

For originally numerical attributes, discretize into bins. Store bin metadata so generated synthetic data can later be mapped back to representative numeric values.

### 4.2 Schema object

Create `schema.py`:

```python
@dataclass
class ColumnSchema:
    name: str
    kind: Literal["categorical", "numerical_binned"]
    cardinality: int
    categories: list[str] | None = None
    bin_edges: list[float] | None = None

@dataclass
class TableSchema:
    columns: list[ColumnSchema]
    label_column: str | None = None

    @property
    def d(self) -> int: ...
    @property
    def cardinalities(self) -> np.ndarray: ...
```

### 4.3 Preprocessing rules

1. Categorical columns:
   - Map each category to an integer ID.
   - Reserve an `UNK` category only if necessary.

2. Numerical columns:
   - Use quantile bins or fixed-width bins.
   - Default: `num_bins=32` for Adult/ACS-like datasets.
   - Encode bin IDs as integers.

3. Missing values:
   - Treat missing as a separate category/bin.

4. Output:
   - Save `schema.json`.
   - Save encoded real data as `encoded.parquet` or `encoded.npy`.

### 4.4 Do not rely on raw floating point features inside QDTE

The first full implementation should operate on finite discrete domains. Continuous halfspace support can be added later by using encoded numerical bin centers, but the default mutation/materialization must keep all rows valid under the schema.

---

## 5. Query representation

QDTE works with Boolean summation queries:

\[
q_j(D) = \sum_{x \in D} \phi_j(x), \quad \phi_j(x) \in \{0,1\}.
\]

Define residual:

\[
r_j = \bar a_j - q_j(\hat D).
\]

Sign convention:

```text
r_j > 0: synthetic data has too few records satisfying query j; candidate edits should make records ENTER query j.
r_j < 0: synthetic data has too many records satisfying query j; candidate edits should make records EXIT query j.
```

### 5.1 Query types

Implement the following query families first:

1. `categorical`: one attribute equals one value.
2. `kway_categorical`: conjunction of several categorical equalities.
3. `range`: one binned numerical attribute lies in `[lo, hi]`.
4. `prefix`: one binned numerical attribute satisfies `value <= threshold`.
5. `mixed`: conjunction of categorical equalities and binned numerical ranges/prefixes.
6. `halfspace`: optional; evaluate on encoded/bin-center features. Keep this as a second-stage feature.

### 5.2 Fixed-shape query catalogue for JAX

JAX likes static shapes. Create a packed query catalogue:

```python
@dataclass
class QueryCatalogue:
    # total number of queries
    m: int

    # query type per query; int32 enum
    type_id: np.ndarray  # [m]

    # up to max_terms conditions per query
    attrs: np.ndarray    # [m, max_terms], int32, -1 padded
    ops: np.ndarray      # [m, max_terms], int32 enum: EQ, LE, GE, RANGE
    values: np.ndarray   # [m, max_terms], int32, for EQ/LE/GE
    lows: np.ndarray     # [m, max_terms], int32, for RANGE
    highs: np.ndarray    # [m, max_terms], int32, for RANGE
    num_terms: np.ndarray # [m], int32

    # optional names/debug metadata stored on CPU
    names: list[str]
    groups: list[str]
```

Each query is a conjunction of conditions. This single representation can cover categorical, k-way categorical, range, prefix, and mixed queries.

### 5.3 Query evaluation kernel

Implement in `queries/eval_jax.py`:

```python
@jax.jit
def eval_records_queries(X_batch, qcat) -> bool_[B, m]:
    """Evaluate all queries on a batch of records.

    X_batch: int32 [B, d]
    qcat fields are device arrays.
    Return B x m Boolean matrix where out[i, j] = phi_j(X_batch[i]).
    """
```

Implementation idea:

1. For each query condition slot `t`, gather `X_batch[:, attrs[:, t]]` to get `[B, m]` values.
2. Evaluate condition depending on `ops`.
3. Mask padded terms.
4. AND across terms.

Pseudo-code:

```python
satisfied = jnp.ones((B, m), dtype=bool)
for term in range(max_terms):
    attr = attrs[:, term]       # [m]
    op = ops[:, term]           # [m]
    valid = attr >= 0           # [m]

    xvals = X_batch[:, attr_clipped]  # [B, m]

    cond_eq = xvals == values[:, term]
    cond_le = xvals <= values[:, term]
    cond_ge = xvals >= values[:, term]
    cond_range = (xvals >= lows[:, term]) & (xvals <= highs[:, term])

    cond = select_by_op(op, cond_eq, cond_le, cond_ge, cond_range)
    cond = jnp.where(valid[None, :], cond, True)
    satisfied = satisfied & cond
return satisfied
```

This dense evaluation is the default backend because it is simple, exact, and likely fast enough on 4090 GPUs for moderate `m`.

### 5.4 Dataset query answers

Implement:

```python
@jax.jit
def answer_queries_shard(X_shard, qcat, batch_size_records=8192) -> float32[m]:
    """Sum phi_j over one shard of records."""
```

Then sum over shards/devices.

For multi-GPU, shard rows across GPUs:

```python
X_sharded = split X into [num_devices, shard_N, d]
partial_counts = pmap(answer_queries_shard)(X_sharded, qcat_replicated)
counts = partial_counts.sum(axis=0)
```

---

## 6. Workload construction

Implement `queries/workload.py` with configurable workload generation.

### 6.1 Default workload for full run

Given a schema, create:

1. One-way categorical queries for every categorical/binned attribute.
2. Two-way categorical marginals for selected pairs, bounded by `max_2way_cells`.
3. Prefix queries for each binned numerical attribute.
4. Range queries for each binned numerical attribute using random or grid intervals.
5. Mixed queries: categorical condition + numerical prefix/range condition.

Example YAML:

```yaml
workload:
  include_oneway: true
  include_2way_cat: true
  include_prefix: true
  include_range: true
  include_mixed: true
  include_halfspace: false
  max_queries: 10000
  max_terms: 4
  numerical_bins: 32
  range_intervals_per_num_attr: 64
  mixed_queries_per_pair: 64
  random_seed: 0
```

### 6.2 Query grouping for DP measurement

Implement two measurement modes:

#### Mode A: `static_all`

Measure all constructed queries in one or more vector groups.

This is simplest and suitable for first full QDTE experiments.

#### Mode B: `adaptive_select_measure`

Implement original select-measure-generate loop later:

1. Construct candidate vector groups.
2. In each round, compute DP exponential mechanism quality score.
3. Select a group.
4. Measure selected group with Gaussian noise.
5. Run QDTE generation with accumulated measurements.

For the first runnable full system, implement `static_all` first, then add adaptive selection.

### 6.3 Sensitivity policy

For a vector query group `V`, use sensitivity:

```text
Delta(V) = 1 if the group is mutually exclusive / orthogonal.
Delta(V) = sqrt(k) if each record can satisfy at most k queries in the group.
Delta(V) = sqrt(|V|) as a safe fallback.
```

For static dense workload where queries overlap heavily, the safe fallback can be too noisy. Therefore, the first full experiment should group queries by orthogonal partitions where possible:

1. One-way categorical full partition: sensitivity 1.
2. One numerical discretized histogram partition: sensitivity 1.
3. Prefix/range/mixed overlapping groups: use bounded group size and conservative sensitivity, or measure them under a separate budget with declared sensitivity.

This code must log the sensitivity used for every measurement group.

---

## 7. Privacy and real DP measurement

### 7.1 zCDP Gaussian mechanism

Gaussian mechanism:

\[
\tilde a = q(D) + \sigma \Delta q \cdot Z, \quad Z \sim \mathcal N(0, I).
\]

If a Gaussian mechanism uses noise standard deviation `sigma * Delta`, it satisfies:

\[
\rho = \frac{1}{2\sigma^2}.
\]

Equivalently, for a group budget \(\rho_g\):

\[
\sigma_g = \frac{1}{\sqrt{2\rho_g}}, \quad
\text{noise std} = \Delta_g \sigma_g.
\]

### 7.2 Measurement module

Implement `measurement/measure.py`:

```python
@dataclass
class MeasurementGroup:
    query_indices: np.ndarray
    sensitivity_l2: float
    rho: float
    sigma: float
    noise_std: float
    name: str

@dataclass
class Measurements:
    target_noisy: np.ndarray       # [m]
    target_projected: np.ndarray   # [m]
    true_answers_debug: np.ndarray | None  # [m], saved only if allowed
    variances: np.ndarray          # [m]
    inv_variances: np.ndarray      # [m]
    groups: list[MeasurementGroup]
    mode: Literal["dp", "oracle"]
```

Default `mode="dp"`:

```python
true_answers = evaluate_queries(X_real, queries)
for each measurement group g:
    sigma = 1 / sqrt(2 * rho_g)
    noise_std = sensitivity_l2_g * sigma
    noisy[group] = true_answers[group] + Normal(0, noise_std)
    variance[group] = noise_std ** 2
```

Debug `mode="oracle"`:

```python
noisy = true_answers
variance = small constant, e.g. 1.0
```

The code must print a warning if `mode="oracle"`:

```text
WARNING: oracle mode uses exact real query answers and is not differentially private. Do not use for paper DP results.
```

### 7.3 Budget config

YAML example:

```yaml
privacy:
  mode: dp                 # dp or oracle
  rho_total: 1.0
  delta: 1.0e-9
  measurement_allocation:
    oneway: 0.25
    twoway: 0.25
    prefix: 0.15
    range: 0.15
    mixed: 0.20
  adaptive_selection: false
```

Ensure allocation sums to 1.0, then group budgets are:

```python
rho_family = rho_total * allocation[family]
rho_group = rho_family / number_of_groups_in_family
```

### 7.4 Conversion to epsilon for logging

For interpretability, log:

\[
\epsilon = \rho + 2\sqrt{\rho \log(1/\delta)}.
\]

---

## 8. Consistency projection / target cleanup

Implement a simple but useful version first.

### 8.1 Projection for partition groups

For every full partition group such as one-way categorical or numerical-bin histogram:

Input noisy counts `y` with length `K`, desired total `N_syn`.

Project onto simplex:

\[
z = \arg\min_{z \ge 0, \sum_i z_i = N} \|z - y\|_2^2.
\]

Implement standard Euclidean simplex projection.

### 8.2 Generic cleanup for non-partition queries

For range/prefix/mixed queries in first implementation:

```python
target_projected[j] = clip(noisy[j], 0, N_syn)
```

Do not attempt full global consistency projection initially. It is expensive and not necessary for first full run.

### 8.3 Optional prefix monotonicity

For prefix queries on the same attribute, enforce monotonicity using isotonic regression or a simple cumulative fix:

```text
if thresholds are increasing, projected prefix counts should be nondecreasing.
```

This is optional for first full run.

---

## 9. Initial synthetic dataset

Initialize `X_syn` from projected one-way marginals.

Default:

```python
for each attribute a:
    p_a = projected_oneway_counts[a] / N_syn
    X_syn[:, a] ~ Categorical(p_a)
```

This gives a valid independent-column baseline and avoids using non-private real rows.

Optional debug initialization:

1. Uniform over domains.
2. Random rows from real data. This is not DP unless using public data, so only for debugging.

---

## 10. QDTE objective and edit advantage

Measured loss:

\[
L(\hat D) = \frac{1}{2}\sum_{j=1}^m \frac{r_j^2}{\sigma_j^2},
\quad r = \bar a - Q(\hat D).
\]

For an edit \(e: x \to x'\), query impact:

\[
\Delta_j(e) = \phi_j(x') - \phi_j(x) \in \{-1,0,1\}.
\]

After applying edit:

\[
r' = r - \Delta(e).
\]

Exact loss change:

\[
\Delta L(e) = -\sum_j \frac{r_j\Delta_j(e)}{\sigma_j^2}
+ \frac{1}{2}\sum_j \frac{\Delta_j(e)^2}{\sigma_j^2}.
\]

Define edit advantage:

\[
A(e) = \sum_j \frac{r_j\Delta_j(e)}{\sigma_j^2}
- \frac{1}{2}\sum_j \frac{\Delta_j(e)^2}{\sigma_j^2}
- \lambda c(e).
\]

If `lambda_cost = 0`, then:

```text
A(e) > 0  <=>  measured loss decreases.
```

Use this formula exactly in code.

### 10.1 Dense GPU scoring backend

Given candidate old rows `[B, d]` and new rows `[B, d]`:

```python
phi_old = eval_records_queries(old_rows, qcat)  # bool [B, m]
phi_new = eval_records_queries(new_rows, qcat)  # bool [B, m]
delta = phi_new.astype(int8) - phi_old.astype(int8)
linear = delta @ (residual * inv_variance)
quad = (delta * delta) @ inv_variance
advantage = linear - 0.5 * quad - lambda_cost * edit_cost
```

This is exact for all measured queries and handles overlapping queries automatically.

### 10.2 Multi-GPU scoring

Split candidate batch across devices:

```text
candidate batch B_total = B_per_device * num_devices
old_rows_sharded: [num_devices, B_per_device, d]
new_rows_sharded: [num_devices, B_per_device, d]
```

Use `jax.pmap(score_candidates_shard)` to compute advantages per device.

Gather advantages and choose best candidates on host or on device.

### 10.3 Compiled-oracle backend, optional later

Add later if dense scoring is slow. For now, keep `score_backend: dense_gpu` as default.

---

## 11. Active query scheduler

Each iteration selects target queries with large residuals.

Define priority:

\[
p_j = \frac{(|r_j| - \kappa \sigma_j)_+}{\sigma_j} + \alpha d_j + \beta w_j.
\]

Where:

- \(\kappa\): noise threshold, default 1.0.
- \(d_j\): debt from previous collateral damage, default 0 initially.
- \(w_j\): optional user importance weight, default 0.

Implementation:

```python
active_mask = abs(residual) > kappa * sigma
priority = where(active_mask, (abs(residual) - kappa * sigma) / sigma + alpha * debt + beta * weight, -inf)
target_query_ids = top_k(priority, num_active_targets)
```

Default parameters:

```yaml
evolution:
  kappa_noise: 1.0
  num_active_targets: 64
  debt_alpha: 0.0      # enable later
  importance_beta: 0.0
```

---

## 12. Query-specific repair / candidate edit generation

QDTE differs from random GA because each candidate edit is generated to enter or exit a selected target query.

For target query `j`:

```text
if residual[j] > 0:
    need ENTER: choose rows with phi_j(x)=0 and generate x' with phi_j(x')=1
if residual[j] < 0:
    need EXIT: choose rows with phi_j(x)=1 and generate x' with phi_j(x')=0
```

### 12.1 Candidate batch design

Generate a large batched set of candidate edits per QDTE iteration:

```yaml
evolution:
  candidates_per_target: 64
  num_active_targets: 64
  total_candidates_per_iter: 4096
  accepted_per_iter: 64
  random_candidate_fraction: 0.05
```

With 2 GPUs, score 4096 to 16384 candidates per iteration if memory allows.

### 12.2 Selecting source rows

Avoid expensive exact `Sat[j]` index lists initially.

For each target query:

1. Sample `over_sample_factor * candidates_per_target` random rows.
2. Evaluate only the target query on those rows.
3. Keep rows matching the needed source state.
4. If too few rows match, sample again or fall back to random rows.

This is simple and works because N is only tens of thousands.

Later optimization: maintain bitset/cache or row lists for active queries.

### 12.3 Repair logic for conjunction queries

Every query is a conjunction of conditions. To ENTER a query, make all conditions true. To EXIT a query, break at least one condition.

#### ENTER repair

For each condition:

- `EQ attr value`: set `x[attr] = value`.
- `LE attr value`: set `x[attr] = min(x[attr], value)`.
- `GE attr value`: set `x[attr] = max(x[attr], value)`.
- `RANGE attr [lo, hi]`: set `x[attr] = clip(x[attr], lo, hi)`.

This produces a minimal deterministic repair.

Add randomized repair variants:

- For EQ, same deterministic value.
- For RANGE, sample uniformly from `[lo, hi]` or from the current synthetic conditional distribution.
- For PREFIX/LE, sample from `[0, threshold]`.

#### EXIT repair

Break one condition, preferably the cheapest one.

For each condition, generate a candidate that violates it:

- `EQ attr value`: set `x[attr]` to a different valid value.
- `LE attr value`: set `x[attr] = min(value + 1, cardinality[attr]-1)` if possible.
- `GE attr value`: set `x[attr] = max(value - 1, 0)` if possible.
- `RANGE attr [lo, hi]`: move to nearest valid value outside the interval if possible.

For each possible break, compute edit cost and optionally score later. Keep 1 to 3 variants.

### 12.4 Edit cost

Default edit cost:

\[
c(e) = \sum_a 1[x_a \ne x'_a] + \gamma \sum_{a \in numerical} |x_a - x'_a| / (K_a - 1).
\]

YAML:

```yaml
evolution:
  lambda_cost: 0.01
  numerical_distance_gamma: 0.1
```

Set `lambda_cost=0.0` for initial theorem/debug runs.

### 12.5 Validity

Every generated `x_new` must satisfy:

```text
0 <= x_new[attr] < cardinality[attr]
```

Add assertions in debug mode.

---

## 13. Candidate object and batch arrays

Avoid Python objects in inner loops. Use arrays.

Candidate batch fields:

```python
row_ids: int32 [B]
old_rows: int32 [B, d]
new_rows: int32 [B, d]
target_query_ids: int32 [B]
edit_cost: float32 [B]
repair_type: int32 [B]
```

For logs only, create Python dataclass after selecting accepted edits.

---

## 14. Transport / edit acceptance

### 14.1 Greedy micro-batch transport

After scoring all candidates:

1. Filter `advantage > min_advantage`.
2. Sort candidates by advantage descending.
3. Select up to `accepted_per_iter` candidates with unique `row_id`.
4. Apply them to `X_syn`.
5. Update residual by the sum of accepted deltas.

Pseudo-code:

```python
for iteration in range(max_iters):
    target_query_ids = scheduler.select(residual, sigma, debt)
    candidates = generate_candidates(X_syn, target_query_ids, residual, qcat, rng)
    scores, deltas = score_candidates(candidates, residual, inv_variance, qcat)
    accepted = select_top_nonconflicting(candidates, scores, max_accept=accepted_per_iter)
    if len(accepted) == 0:
        patience += 1
        if patience >= stop_patience:
            break
        continue
    X_syn = apply_edits(X_syn, accepted)
    residual = residual - sum(deltas[accepted], axis=0)
    answer_syn = answer_syn + sum(deltas[accepted], axis=0)
    periodically full recompute answer_syn and residual
```

### 14.2 Exact sequential greedy option

For theoretical monotonicity, implement an option:

```yaml
evolution:
  transport_mode: sequential_greedy
```

This applies only one edit at a time or recomputes advantage after each accepted edit. It is slower but useful for tests.

### 14.3 Fast micro-batch option

Default full run:

```yaml
evolution:
  transport_mode: microbatch_greedy
  accepted_per_iter: 64
```

This is faster but uses stale residual within one micro-batch. Periodic full recomputation catches drift.

### 14.4 Batch sanity check

Before applying a micro-batch, compute cumulative impact:

\[
\Delta_F = \sum_{e \in B} \Delta(e).
\]

Batch advantage:

\[
A(B) = r^T \Sigma^{-1}\Delta_F - \frac12 \Delta_F^T \Sigma^{-1}\Delta_F - \lambda \sum_{e\in B} c(e).
\]

If `A(B) <= 0`, reduce batch size by taking fewer top edits until positive. This prevents over-application.

---

## 15. State and residual updates

Create `evolution/state.py`:

```python
@dataclass
class QDTEState:
    X_syn: jax.Array           # [N_syn, d]
    answer_syn: jax.Array      # [m]
    target: jax.Array          # [m]
    residual: jax.Array        # [m]
    variance: jax.Array        # [m]
    inv_variance: jax.Array    # [m]
    sigma: jax.Array           # [m]
    debt: jax.Array            # [m]
    iteration: int
    rng_key: jax.Array
```

Update residual exactly from accepted deltas:

```python
answer_syn_new = answer_syn + delta_sum
residual_new = target - answer_syn_new
```

Every `full_recompute_every` iterations:

```python
answer_syn = evaluate_queries(X_syn, qcat)
residual = target - answer_syn
```

Default:

```yaml
evolution:
  full_recompute_every: 50
```

If dense scoring is exact and updates are correct, full recompute should match incremental updates up to zero or tiny numerical error.

---

## 16. Dense GPU memory estimates

For candidate scoring:

```text
B = 4096 candidates
m = 10000 queries
phi_old bool [B,m] ≈ 40 MB if bool/int8
phi_new bool [B,m] ≈ 40 MB
delta int8 [B,m] ≈ 40 MB
float matmul may cast to float32, temporary ≈ 160 MB
```

This is acceptable on RTX 4090 if implemented carefully.

For `B=16384`, memory can rise above 1 GB, still possible but tune carefully.

Use chunking if needed:

```yaml
evolution:
  scoring_chunk_size: 4096
```

---

## 17. Multi-GPU implementation details

### 17.1 Candidate scoring with pmap

Implement:

```python
@partial(jax.pmap, in_axes=(0, 0, None, None, None, None))
def score_candidates_pmap(old_rows_shard, new_rows_shard, residual, inv_variance, qcat, edit_cost_shard):
    phi_old = eval_records_queries(old_rows_shard, qcat)
    phi_new = eval_records_queries(new_rows_shard, qcat)
    delta = phi_new.astype(jnp.int8) - phi_old.astype(jnp.int8)
    w = residual * inv_variance
    linear = delta.astype(jnp.float32) @ w.astype(jnp.float32)
    quad = (delta.astype(jnp.float32) * delta.astype(jnp.float32)) @ inv_variance.astype(jnp.float32)
    advantage = linear - 0.5 * quad - lambda_cost * edit_cost_shard
    return advantage, delta
```

Need static lambda or pass `lambda_cost` as an argument.

### 17.2 Shard candidate batch

If there are 2 GPUs and `B_total=8192`, shape as:

```text
old_rows_sharded: [2, 4096, d]
new_rows_sharded: [2, 4096, d]
```

### 17.3 Full answer computation with pmap

Shard `X_syn` or `X_real` by rows:

```python
@jax.pmap
def answer_queries_pmap(X_shard, qcat):
    return answer_queries_shard(X_shard, qcat)

counts = answer_queries_pmap(X_sharded, qcat).sum(axis=0)
```

### 17.4 Do not move large arrays CPU/GPU repeatedly

Keep these on GPU:

- `X_syn`
- query catalogue arrays
- residual/target/variance

Only move small summaries and selected top candidates to CPU for logging.

---

## 18. QDTE engine config

Example `configs/adult_qdte.yaml`:

```yaml
run:
  dataset_name: adult
  input_csv: data/adult.csv
  output_dir: outputs/adult_qdte
  seed: 0
  device: gpu

preprocess:
  numerical_bins: 32
  missing_token: __MISSING__
  label_column: income

workload:
  include_oneway: true
  include_2way_cat: true
  include_prefix: true
  include_range: true
  include_mixed: true
  include_halfspace: false
  max_queries: 10000
  max_terms: 4
  range_intervals_per_num_attr: 64
  mixed_queries_per_pair: 64
  max_2way_cells: 5000
  random_seed: 0

privacy:
  mode: dp
  rho_total: 1.0
  delta: 1.0e-9
  measurement_mode: static_all
  measurement_allocation:
    oneway: 0.25
    twoway: 0.25
    prefix: 0.15
    range: 0.15
    mixed: 0.20

projection:
  project_partitions: true
  clip_nonpartition: true
  prefix_monotonicity: false

init:
  N_syn: same_as_real
  method: independent_oneway

qdte:
  score_backend: dense_gpu
  transport_mode: microbatch_greedy
  max_iters: 5000
  num_active_targets: 64
  candidates_per_target: 64
  total_candidates_per_iter: 4096
  accepted_per_iter: 64
  kappa_noise: 1.0
  lambda_cost: 0.01
  numerical_distance_gamma: 0.1
  random_candidate_fraction: 0.05
  full_recompute_every: 50
  stop_patience: 50
  min_advantage: 1.0e-6
  log_every: 10
  eval_every: 100

runtime:
  use_pmap: true
  scoring_chunk_size: 4096
  xla_preallocate: true

evaluation:
  compute_true_query_error: true     # allowed only for experiments; not used in generation
  downstream_ml: true
  save_synthetic_csv: true
```

---

## 19. Evaluation metrics

Implement `eval/metrics.py`.

### 19.1 Measured loss

\[
L_{measured} = \frac12 \sum_j r_j^2 / \sigma_j^2.
\]

This can be computed during generation.

### 19.2 True query error for experiments only

For evaluation after generation, compute exact true answers on real data:

\[
\text{MAE} = \frac{1}{m}\sum_j |q_j(D)/N - q_j(\hat D)/N|.
\]

\[
\text{RMSE} = \sqrt{\frac{1}{m}\sum_j (q_j(D)/N - q_j(\hat D)/N)^2}.
\]

Important: true query error uses real data and is not part of generation. It is for offline experiment evaluation only.

### 19.3 Runtime metrics

Log every run:

```text
wall_clock_seconds
time_measurement_seconds
time_init_seconds
time_generation_seconds
time_candidate_generation_seconds
time_scoring_seconds
time_transport_seconds
time_full_recompute_seconds
num_iterations
num_candidates_scored
num_accepted_edits
accepted_rate
positive_advantage_rate
mean_advantage_accepted
peak_gpu_memory_if_available
```

### 19.4 Directed diagnostics

```text
target_query_improvement_rate
collateral_damage_mean
batch_advantage
residual_norm_trajectory
active_query_count
query_starvation_rate optional
```

### 19.5 Downstream ML

Use train-on-synthetic, test-on-real.

Implement simple classifiers:

1. Logistic regression.
2. Random forest.
3. SVM if needed.

Use encoded or decoded data consistently.

---

## 20. Smoke tests

Create `scripts/smoke_qdte.py`.

Use a tiny synthetic dataset generated in code:

```text
N = 1000
D = 4
cardinalities = [4, 5, 8, 8]
queries = oneway + prefix + range + mixed
privacy.mode = dp and oracle variants
max_iters = 100
```

Smoke test must verify:

1. DP measurement actually evaluates real `X_real` and adds noise.
2. Oracle mode gives exact target counts.
3. Query evaluation shapes are correct.
4. Every repair produces valid rows.
5. At least some candidates have positive advantage.
6. Measured loss decreases after accepted edits in sequential mode.
7. Full recompute equals incremental residual.
8. Synthetic output file is written.

Run:

```bash
python scripts/smoke_qdte.py --mode dp
python scripts/smoke_qdte.py --mode oracle
pytest -q
```

---

## 21. Unit tests that must pass

### 21.1 Sign convention test

Construct a one-way categorical query target where target count is larger than synthetic count.

Expected:

```text
residual > 0
ENTER edit delta = +1
advantage positive if residual is sufficiently large
```

### 21.2 Edit advantage identity test

For random residual and random candidate edit:

```python
loss_before = 0.5 * sum(residual**2 * inv_var)
residual_after = residual - delta
loss_after = 0.5 * sum(residual_after**2 * inv_var)
adv_no_cost = sum(residual * delta * inv_var) - 0.5 * sum(delta**2 * inv_var)
assert allclose(loss_after - loss_before, -adv_no_cost)
```

### 21.3 Repair success tests

For each query type:

- Generate records not satisfying query.
- Apply ENTER repair.
- Verify query satisfaction becomes 1.
- Generate records satisfying query.
- Apply EXIT repair.
- Verify query satisfaction becomes 0.

### 21.4 Measurement noise test

For a group with `rho_g`, verify:

```python
sigma = 1 / sqrt(2 * rho_g)
noise_std = sensitivity * sigma
variance = noise_std ** 2
```

### 21.5 Batch transport sanity

If `transport_mode=sequential_greedy` and `lambda_cost=0`, every accepted edit must reduce measured loss.

---

## 22. Full run commands

### 22.1 Adult full run

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py \
  --config configs/adult_qdte.yaml \
  --privacy.mode dp \
  --run.seed 0
```

### 22.2 Oracle upper-bound run

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py \
  --config configs/adult_qdte.yaml \
  --privacy.mode oracle \
  --run.seed 0
```

Again, oracle mode is not DP. It is useful to test optimization capacity without DP noise.

### 22.3 Ablation runs

```bash
# Random mutation baseline inside same engine
python scripts/run_ablation.py --config configs/adult_qdte.yaml --variant random_mutation

# QDTE without cost penalty
python scripts/run_ablation.py --config configs/adult_qdte.yaml --variant no_edit_cost

# Sequential exact greedy
python scripts/run_ablation.py --config configs/adult_qdte.yaml --variant sequential_greedy

# No noise threshold
python scripts/run_ablation.py --config configs/adult_qdte.yaml --variant no_threshold
```

---

## 23. Internal baselines to implement immediately

Even before connecting external Private-GSD code, implement internal baselines under the same query evaluator:

### 23.1 Random mutation baseline

At each iteration:

1. Randomly sample rows.
2. Randomly change one or more attributes.
3. Score by exact edit advantage.
4. Accept if positive.

This tests whether query-directed candidate generation matters.

### 23.2 Target-only QDTE

Generate query-directed edits but score only the target query, ignoring collateral effects.

This tests whether global edit-impact scoring matters.

### 23.3 Full QDTE

Generate query-directed edits and score against all measured queries.

---

## 24. Private-GSD comparison wrapper

If existing Private-GSD code is available, do not rewrite it first. Add a wrapper script that:

1. Uses the same preprocessed data.
2. Uses the same measured workload and DP targets if possible.
3. Runs Private-GSD with its original optimizer.
4. Exports synthetic data in the same format.
5. Evaluates with the same `eval/metrics.py`.

The first QDTE implementation should focus on making QDTE correct and fast. Private-GSD wrapper can be added once QDTE full run works.

---

## 25. Logging and outputs

Each run writes:

```text
outputs/<run_id>/
  config_resolved.yaml
  schema.json
  measurements.json
  queries.parquet or queries.json
  synthetic_encoded.npy
  synthetic_decoded.csv
  metrics_final.json
  metrics_timeseries.csv
  runtime.json
  logs.txt
```

`metrics_timeseries.csv` columns:

```text
iteration
wall_time
measured_loss
residual_l2
residual_l1
active_queries
num_candidates
positive_advantage_rate
accepted_edits
accepted_rate
mean_advantage
batch_advantage
true_query_mae optional
true_query_rmse optional
```

---

## 26. Implementation order for Codex

Do not implement everything at once. Implement in this order.

### Step 1: Repo skeleton and config loader

Deliver:

- `qdte/config.py`
- YAML loading and override support.
- `scripts/check_env.py`.

### Step 2: Schema and preprocessing

Deliver:

- CSV loading.
- categorical encoding.
- numerical binning.
- `schema.json` saving.
- encoded data saving/loading.

### Step 3: Query catalogue and dense JAX query evaluator

Deliver:

- `QueryCatalogue`.
- workload builder for one-way, prefix, range, mixed.
- `eval_records_queries`.
- `answer_queries` with batching.
- tests for query correctness.

### Step 4: Real DP measurement

Deliver:

- `measure_real_dataset_dp(X_real, qcat, groups, rho_total, config)`.
- Gaussian zCDP noise.
- `oracle` mode with explicit warning.
- measurement variance and inverse variance.
- partition projection.

This step must not use simulated targets except in tests.

### Step 5: Initial synthetic data

Deliver:

- independent one-way initialization from projected one-way marginals.
- valid integer-coded `X_syn`.

### Step 6: Candidate repair generation

Deliver:

- target query scheduler.
- source-row sampling.
- ENTER/EXIT repairs for conjunction queries.
- candidate batch arrays.
- tests for repair success.

### Step 7: Dense edit-advantage scoring

Deliver:

- candidate scoring using exact formula over all measured queries.
- pmap candidate scoring if 2 GPUs available.
- edit advantage identity test.

### Step 8: Greedy and micro-batch transport

Deliver:

- select top non-conflicting candidates.
- batch advantage sanity check.
- apply edits.
- residual update.
- periodic full recomputation.

### Step 9: QDTE engine

Deliver:

- end-to-end generation loop.
- metrics logging.
- stop conditions.

### Step 10: Smoke and full scripts

Deliver:

- `scripts/smoke_qdte.py`.
- `scripts/run_qdte.py`.
- `configs/smoke.yaml`.
- `configs/adult_qdte.yaml`.

### Step 11: Evaluation metrics

Deliver:

- query MAE/RMSE/MaxErr.
- measured loss.
- runtime metrics.
- downstream ML optional.

### Step 12: Ablations

Deliver:

- random mutation baseline.
- target-only QDTE.
- sequential vs microbatch.
- oracle vs dp.

---

## 27. Performance targets

The first full implementation should aim for:

```text
Candidate scoring throughput: >= 50,000 candidates/sec for m <= 10,000 if B is large enough.
Full answer recompute: <= a few seconds for N <= 50,000 and m <= 10,000.
Adult full QDTE run: <= 1 hour for max_iters around 5,000, preferably much less.
```

These are targets, not hard requirements. Log actual throughput.

---

## 28. Common pitfalls to avoid

1. Do not use non-private exact real query answers during generation in DP mode.
2. Do not silently switch to oracle mode.
3. Do not use inconsistent residual sign.
4. Do not update residual with the wrong sign. If `answer_syn += delta`, then `residual -= delta`.
5. Do not implement candidate scoring only on target query for the full method. Full QDTE must score all measured queries.
6. Do not allow invalid category/bin values after repair.
7. Do not move huge candidate matrices to CPU each iteration.
8. Do not evaluate true query error during generation unless explicitly configured for diagnostics; even then, do not use it for decisions.
9. Do not claim DP if `privacy.mode=oracle`.
10. Do not make halfspace the bottleneck of the first full implementation.

---

## 29. Definition of done

The implementation is complete when the following command runs end-to-end and writes synthetic data plus metrics:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
```

It must produce:

```text
synthetic_decoded.csv
metrics_final.json
metrics_timeseries.csv
runtime.json
```

And the logs must show:

```text
- real DP measurement performed on X_real
- rho_total and epsilon(delta) logged
- number of measured queries
- initial measured loss
- final measured loss
- true query error for evaluation, if enabled
- number of candidates scored
- number of accepted edits
- runtime breakdown
- GPU device count
```

---

## 30. Short implementation prompt for Codex

Use this prompt when giving the plan to Codex:

```text
Implement the QDTE algorithm exactly according to QDTE_FULL_IMPLEMENTATION_PLAN_FOR_CODEX.md.
The implementation must be real end-to-end DP synthetic tabular data generation, not a simulated-target demo.
Use JAX as the core computational framework.
Default measurement mode must compute query answers on the real input dataset and add Gaussian zCDP noise.
Oracle mode is allowed only for debugging and must print a non-DP warning.
Use dense GPU exact edit-impact scoring first: evaluate all measured queries on old/new candidate rows and compute the edit advantage formula exactly.
Implement one-way/prefix/range/mixed query workloads, real DP measurement, independent one-way initialization, query-directed candidate repair, micro-batch greedy transport, residual updates, periodic full recomputation, metrics, and smoke tests.
The final command must run:
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_qdte.py --config configs/adult_qdte.yaml --privacy.mode dp
and produce a decoded synthetic CSV plus metrics and runtime logs.
```
