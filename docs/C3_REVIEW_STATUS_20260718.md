# C3-CDWF 公开审查状态

日期：2026-07-18

方法：`SAGE-QDTE-RCE-C3-CDWF-v1`

用途：给外部专家提供可复核的当前代码状态和唯一剩余机制问题。

## 0. RHCG-CCMP v2 更新（当前状态）

本节取代下方旧 restricted-hull 状态作为当前决策入口；旧章节保留用于说明
问题如何从 finite dictionary coverage 推进到 full-row pricing tractability。

专家选择 released-history column generation 后，公开分支现已实现：

```text
sealed public legal-row domain
unary/pairwise canonical shadow map
Phase-I global row-realizable margin
minimal confidence inflation
Phase-II CCF moment projection
minimum-norm dual
exact legal-row MAP pricing
global Frank-Wolfe certificate
```

Stage 0 exhaustive certification通过。Adult、epsilon=0.1、no-truth Stage 1 的
首个 master也已通过：

```text
s_star:                -6.8405247478e-06
inflation:              0
active face:            global ellipsoid
master runtime:         0.0234 seconds
master dual:            certified
```

因此旧的 31-table coverage问题已解决。当前新阻塞是完整 105-pair
categorical MAP pricing：

```text
HiGHS:  roughly 3 minutes without a global certificate
SCIP:   119.46 seconds, remaining gap 42.85%
public branching / binary-pair / triangle-cut variants:
        no paper-scale exact certificate
```

按照冻结协议，uncertified pricing是 computational fallback，所以 Stage 1、
formal seeds、最终 QDTE、true evaluator和 AIM panel均未获准启动。该结果不表示
C3 utility输给 AIM，也不表示 full row polytope与 confidence set不相交。

完整实现结果、数值和下一项四选一决策见：

```text
docs/SAGE_QDTE_RCE_C3_RHCG_CCMP_V1_专家决断与实施计划_20260718.md
```

建议专家只冻结一个 tractability boundary：

```text
A. bounded-treewidth exact shadow
B. certified nonzero pricing interval
C. separable/blockwise exact pressure
D. stop RHCG-CDWF for the current paper
```

## 1. 当前结论

`C3-CDWF` 的预算拆分、streamwise Gaussian transcript、sequential CCF、
restricted relaxed RCE、minimum-norm dual、confidence pressure、dual
water-filling、zCDP ledger 和 fail-closed 路径均已实现。

当前尚未完成正式实验，也不能声称打赢 AIM。开发格首先暴露的问题是：

> 随 refinement observations 增加，streamwise confidence set 收紧并移动；
> 仅根据 base transcript 封存的有限 empirical-table convex hull，无法保证
> 在后续轮次仍与该 confidence set 相交。

因此当前阻塞位于 restricted dictionary coverage，不是 privacy ledger、
water-filling 公式或 QDTE 最终生成阶段。

## 2. 已实现的冻结协议

预算仅在 eligible pure-interaction blocks 内重分配：

```text
interaction base rho:       (25/36) * control interaction rho
interaction refinement rho: (11/36) * control interaction rho
one-way/noneligible rho:    unchanged from control
epsilon=0.1 rounds:         4
epsilon=0.3 rounds:         6
per-block cap:              final rho <= 4 * control rho
selection privacy cost:     0
```

每轮执行：

```text
1. precision-combine released Gaussian observations
2. update transcript-only sequential CCF
3. solve restricted relaxed RCE
4. certify the minimum-L2-norm restricted dual
5. compute blockwise confidence pressure
6. solve deterministic dual water-filling
7. release and charge independent Gaussian refinements
```

任何 dual certificate failure 均 fail closed 到 public-uniform allocation；正式
cell 只要出现一次 fallback，就不具备 promotion eligibility。

## 3. 当前开发证据

开发设置：Adult、epsilon=0.1、DualWaterFill、四轮、全程不加载 true utility。
base transcript 构造的 restricted dictionary 包含 3 个 released-only anchors 和
28 个 QDTE path snapshots，共 31 张经验表。

```text
round 0:
  restricted dual certified: yes
  dual fallback:             no
  refined blocks:            19 / 105

round 1:
  restricted dual certified: yes
  dual fallback:             no
  refined blocks:            33 / 105
  relative primal-dual gap:  1.1316e-8
  stationarity residual:     0
  complementarity residual:  8.4674e-8

round 2:
  minimum required face slack: 0.1657838256
  dual fallback:               yes

round 3:
  minimum required face slack: 0.2339208604
  dual fallback:               yes
```

Round 2 的数值是独立 stage-one convex feasibility gap：即使忽略 KL objective，
当前 31-table hull 中也不存在满足该轮 confidence constraints 的 convex
combination。因此它不是通过放宽 SLSQP success 判定即可修复的数值误差。

已尝试且撤回：

```text
snapshot interval 16 -> 8:
  dictionary tables 31 -> 55
  round-2 slack 0.16578 -> 0.16573

continue the same noisy-target path after first feasibility:
  did not restore later-round coverage
  reduced early certificate stability
```

这些结果只否定“沿同一 base noisy-target path 加密或加深快照”这一修补，
不否定 full row-realizable RCE、CCF、QDTE 或 interaction-precision ceiling。

## 4. 需要专家拍板的唯一问题

请在以下三项中冻结一项：

```text
A. frozen_dictionary
   在 refinement 前，仅用 base DP transcript 和 public metadata，构造并封存
   一个具有明确规模上限的 confidence-cover dictionary。

B. released_column_generation
   每轮仅用当时 released history 做 certificate-qualified column generation，
   再在扩充后的声明模型上求 minimum-norm restricted dual。

C. stop_c3_v1
   保持当前 frozen empirical-table hull 和 zero-fallback gate，并判定 C3-v1
   不进入 formal panel。
```

若选择 A，请同时冻结输入、随机种子、路径数、每条路径预算、snapshot 规则、
table/cell/memory cap、tie-break 和无法覆盖时的 fallback。

若选择 B，请同时冻结：

```text
mathematical object represented by each column
pricing objective and deterministic oracle
per-round and total column caps
reduced-cost / feasibility certificate thresholds
dictionary sharing rules for SplitUniform and DualWaterFill
fallback semantics when the cap is exhausted
```

建议的下一次最小 smoke 是：

```text
dataset: Adult
epsilon: 0.1
seed: development seed only
gate: four rounds, zero fallback
generator/evaluator: disabled
```

只有该 no-truth mechanism gate 通过后，才应扩到 BR2000；二者通过后再封存
formal seeds 并运行最终 QDTE/AIM panel。

## 5. 代码审查入口

核心实现：

```text
qdte/measurement/cdwf.py
qdte/measurement/cdwf_protocol.py
qdte/measurement/cdwf_transcript.py
qdte/measurement/adaptive_interactions.py
qdte/rce/cdwf_shadow.py
qdte/rce/cdwf_dual.py
qdte/rce/streamwise.py
qdte/rce/streamwise_dual.py
qdte/rce/sequential_forest_prior.py
scripts/run_rce_c3_cdwf_measurement_cell.py
```

Focused tests：

```text
tests/test_cdwf.py
tests/test_cdwf_protocol.py
tests/test_cdwf_transcript.py
tests/test_cdwf_engine.py
tests/test_rce_cdwf_dual.py
tests/test_rce_cdwf_shadow.py
tests/test_rce_streamwise.py
tests/test_rce_streamwise_dual.py
tests/test_rce_sequential_forest_prior.py
tests/test_rce_relaxed.py
```

公开快照不包含 private data、raw outputs、内部 handoff 或离线 true evaluator
artifact。开发数值在本文中以必要的聚合结果记录。

## 6. 当前验证

```text
C3 focused tests: 30 passed
public branch full tests: 248 passed, 1 skipped
compileall:        passed
git diff --check: passed
```

推荐复核命令：

```bash
conda run -n qdte python -m pytest -q \
  tests/test_cdwf.py \
  tests/test_cdwf_protocol.py \
  tests/test_cdwf_transcript.py \
  tests/test_cdwf_engine.py \
  tests/test_rce_cdwf_dual.py \
  tests/test_rce_cdwf_shadow.py \
  tests/test_rce_streamwise.py \
  tests/test_rce_streamwise_dual.py \
  tests/test_rce_sequential_forest_prior.py \
  tests/test_rce_relaxed.py
```

## 7. 声明边界

当前可以说：

```text
CDWF 的 released-only pressure 和非均匀 water-filling 已在前两轮工作；
当前 frozen restricted dictionary 不足以覆盖后续收紧的 confidence geometry。
```

当前不能说：

```text
C3-CDWF 已打赢 AIM；
formal promotion gate 已通过；
full row-realizable confidence set 不可行；
QDTE generator 已被该开发格否证。
```
