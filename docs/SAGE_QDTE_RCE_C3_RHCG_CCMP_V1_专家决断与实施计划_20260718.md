# SAGE-QDTE-RCE C3 RHCG-CCMP-v1 专家决断与实施计划

日期：2026-07-18

状态：专家已授权实现，先通过 exhaustive toy certification，再运行 no-truth
Adult mechanism smoke。

## 1. 冻结决断

```text
decision:
  released_history_column_generation_v2

parent_method:
  SAGE-QDTE-RCE-C3-CDWF-v1

new_subroutine:
  RHCG-CCMP-v1

full_name:
  Released-History Column Generation
  with Certified CCF Moment Projection
```

不允许继续使用：

```text
first-feasible restricted empirical-table hull
  -> restricted minimum-KL dual
  -> C3 pressure
```

每轮 allocation shadow oracle 改为：

```text
global row-polytope confidence margin
  -> minimal reported confidence inflation
  -> certified CCF moment projection
  -> globally priced minimum-norm confidence dual
  -> unchanged logarithmic water-filling
```

最终 estimator/generator 仍使用原 CCF-RCE-QDTE minimum-KL 路径。P2 只是
allocation surrogate，不能冒充 final RCE。

## 2. 公共合法行域

冻结公共域：

$$
\mathcal X_{\rm pub}=\prod_j\{0,\ldots,d_j-1\},
$$

除非另有显式 public row constraints。formal path 必须在 private measurement
前封存：

```text
public_schema.json
public_legal_row_manifest.json
attribute/category ordering
missing-value rule
public n
schema SHA-256
```

schema 缺失、从 private CSV 推断 cardinality、或存在未实现的 public row
constraint 时 fail closed。

## 3. Shadow feature map

`A_shadow` 只允许 exact unary/pairwise row energies：

```text
one-way contrast blocks
pure pairwise interaction blocks
明确编译成 unary/pairwise energy 的 derived factors
```

以下情况 preflight failure：

```text
factor order > 2
multi-attribute halfspace
不能复用 canonical evaluator feature 的 query
```

`A_final` 仍可包含完整 released transcript。不能假设完整 workload 都可用于
pairwise pricing。

## 4. Canonical columns 与 cap

新 column 必须是合法 row Dirac atom。现有 31 张 empirical tables 只可作为
aggregate warm-start columns，不是理论原子。禁止新增 QDTE table path、
snapshots、pseudo-target、bootstrap 或 fission dictionary paths。

公开 affine rank：

$$
r_{\rm shadow}=
\sum_j(d_j-1)+
\sum_{(j,k)\in\mathcal E_{\rm shadow}}(d_j-1)(d_k-1).
$$

```text
K_max = min(r_shadow + 1, 4096)
```

`4096` 是 compute cap，不是 theorem。cap exhaustion 是 computational fallback。

## 5. Phase I：全局 row-realizable margin

对 canonical dimensionless constraints：

$$
g_2(p)=\frac{(Ap-\bar z)^T\Omega(Ap-\bar z)}{c_2}-1,
$$

$$
g_q^+(p)=\frac{(Ap-\bar z)_q}{b_q}-1,
\qquad
g_q^-(p)=-\frac{(Ap-\bar z)_q}{b_q}-1,
$$

求：

$$
s^\star=\min_{p\in\Delta(\mathcal X_{\rm pub})}\max_i g_i(p).
$$

`s` 不得约束非负。exact legal-row pricing 和 global FW gap 必须认证到
`1e-8`。

定义：

```text
s_bar = max(0, s_star)
```

`s_bar > 0` 是 row-realizability inflation certificate，不是 fallback。膨胀后的
constraints 为 `g_i(p) <= s_bar`。

## 6. Phase II：Certified CCF Moment Projection

由 released CCF prior 精确计算 shadow moments `m_F,g`。对 block 的 unit-rho
covariance `C_g`，求：

$$
\min_{p\in\Delta(\mathcal X_{\rm pub})}
\frac12\sum_g
(A_gp-m_{F,g})^T C_g^\dagger(A_gp-m_{F,g})
$$

满足：

```text
g_i(p) <= s_bar
```

目标不依赖当前 `rho_g`，不引入 entropy lambda、family weight 或 truth。使用
exact row pricing 和 global FW gap certificate。

## 7. Pricing 与 dual

pricing energy 必须写成 canonical unary/pairwise categorical MAP：

$$
\psi(x)=\sum_j\theta_{j,x_j}+
\sum_{(j,k)}\theta_{jk,x_j,x_k}.
$$

大域使用 exact/certified MILP，小域 exhaustive enumeration。固定：

```text
threads: 1
solver seed: 0
node cap: 2,000,000
pricing gap <= 1e-9 * max(1, abs(incumbent))
lexicographic public-ID tie-break
```

全局 gap：

$$
G=\mathbb E_{X\sim p}[\psi(X)]-
\min_{x\in\mathcal X_{\rm pub}}\psi(x)\le10^{-8}.
$$

dual 来源冻结为：

```text
s_star > 0:
  Phase-I minimum-norm global dual

s_star <= 0:
  Phase-II minimum-norm global dual
```

两种 dual 不混合加权。

## 8. Pressure 与 water-filling

pressure 是 canonical constraints 关于 `log rho_g` 的 envelope derivative：

$$
\Lambda_g=
\left[\frac{\partial\mathcal L}{\partial\log\rho_g}\right]_+.
$$

实现必须同时提供 closed-form 和 autodiff/finite-difference audit。原有
water-filling、`gamma <= 4`、4/6 rounds 和 interaction-only split 保持不变。

## 9. Combined shadow 与 final safe geometry

```text
exact precision-combined geometry:
  only for CCF update surrogate, RHCG-CCMP shadow dual and allocation

alpha-spent streamwise confidence intersection:
  final CCF-RCE-QDTE and coverage theorem
```

不能用 adaptive combined target 写 fixed-design final coverage claim。

## 10. Fallback 定义

不算 fallback：

```text
released-history column generation
新增合法 row atoms
s_star > 0
使用并报告 minimal confidence inflation
不同 released histories 产生不同 dictionary
```

算 fallback并失去 promotion eligibility：

```text
uncertified MILP pricing
uncertified master primal/dual/FW gap
atom cap exhaustion
public schema/factor-order failure
positive gap但 priced row 重复
privacy ledger mismatch
truth/evaluator object loaded
```

## 11. Causal arms

```text
AllStatic-CG
SplitUniform-CG
DualWaterFill-CG
OfficialAIM
```

A/B/C 使用同一个 deterministic history-to-optimizer algorithm；realized
dictionaries 可因 released histories 不同而不同。

## 12. 执行顺序

### Stage 0：exhaustive toy certification

```text
attributes: 3--5
cardinalities: 2--4
legal rows: fully enumerable
```

要求：

```text
Phase-I CG vs enumeration objective error       <= 1e-9
Phase-II CG vs enumeration objective error      <= 1e-9
MILP vs enumeration pricing error               <= 1e-10
global FW gap correctness                       certified
Lambda finite-difference relative error          <= 1e-5
false full-optimality certificates               0
deterministic dictionary hash                    identical
```

### Stage 1：Adult mechanism-only smoke

```text
dataset: Adult
epsilon: 0.1
base seed: 9101
refinement seed: 9202
generation seed: 9303
rounds: 4
truth/evaluator/final QDTE: disabled
```

四轮必须完成并通过所有 global master/pricing certificates；不要求
`s_star <= 0`。

Stage 1 通过后依次运行 BR2000 epsilon=0.1，再运行 Adult/BR2000
epsilon=0.3。四个 development cells 全部通过后才生成 formal seeds 100--104。

## 13. 当前实施边界

先新增 RHCG-CCMP 模块和 toy tests。旧 restricted empirical-table solver保留为
回归对照，不再作为 C3 pressure 的最终来源。Stage 0 全部通过前禁止启动 Adult
private measurement。

## 14. 2026-07-18 实施结果

专家冻结的 RHCG-CCMP 主体已经实现：

```text
public legal-row domain + schema manifest
unary/pairwise canonical shadow feature map
Phase I global row-realizable margin, s unrestricted
minimal confidence inflation, max(0, s_star)
Phase II unit-rho CCF moment projection
minimum-norm dual extraction
exact legal-row MAP/MILP pricing
global Frank-Wolfe gap
combined-shadow / streamwise-final geometry separation
RHCG-backed C3 protocol for both uniform and dual arms
```

旧 released-only GPU directed path 被恢复，但只承担 aggregate warm start：31 张
path tables 中仅保留三个 released anchors 和最终 directed endpoint进入 master；
中间高度相关 snapshots 只作为诊断。global certificate仍只来自 public legal-row
pricing，aggregate warm columns不承担全局最优性 claim。

## 15. Stage 0 结果

Stage 0 已通过：

```text
exhaustive domains:
  (2, 3, 4)
  (2, 2, 3, 2)
  (2, 2, 2, 2, 2)

Phase I objective error:        <= 1e-9
Phase II objective error:       <= 1e-9
MILP pricing error:             <= 1e-10
global gap:                     <= 1e-8
pressure finite-difference:     <= 1e-5 relative
false global certificates:      0
deterministic hashes:           passed
focused verification:          12 passed
```

## 16. Adult Stage 1 当前结果

冻结 cell：

```text
dataset:          Adult
epsilon:          0.1
base seed:        9101
refinement seed:  9202
generation seed:  9303
truth/evaluator:  never loaded
final QDTE:       not run
```

公开规模：

```text
n:                         45,222
attributes:                15
cardinalities:             (16,7,16,16,16,7,14,6,5,2,8,15,16,41,2)
pair blocks:               105
total shadow blocks:       120
shadow feature dimension:  13,337
public row-domain size:    424,688,379,494,400
column cap:                4,096 (compute cap, not theorem-qualified rank cap)
```

GPU warm path执行 391 rounds、生成 31 个 snapshots，用时约 23 秒；保留 4 个
aggregate warm columns。首个 Phase-I master 已成功：

```text
master runtime:            0.0234 seconds
s_star:                   -6.8405247478e-06
inflation:                 0
active confidence face:    global ellipsoid only (constraint 0)
master dual:               certified
```

因此旧的 `31-table hull infeasible` 阻塞已被解决：完整 row-polytope 的当前
warm point具有严格 interior。新的阻塞发生在首个 full-row pricing oracle。

## 17. Full-row pricing 计算阻塞

Adult 的完整 shadow graph含全部 105 个 pair factors。一次 pricing 是：

$$
\min_x\left[\sum_j\theta_j(x_j)+
\sum_{j<k}\theta_{jk}(x_j,x_k)\right].
$$

这是一般 complete-graph categorical pairwise MAP，在一般情形下是 NP-hard。
当前首个 released-only instance 的实测如下：

| Exact backend / strengthening | 结果 |
|---|---|
| HiGHS，185 unary binaries + 15,678 pair variables | 超过约 3 分钟仍未完成 primary global certificate |
| SCIP，同一 continuous-pair formulation | 119.46 秒时 primal `-0.7277391`、dual `-1.0396050`、gap `42.85%` |
| SCIP，pair variables 全部 binary | 60 秒时 gap `479.3%`，更差 |
| public low-cardinality branching，固定 5 attrs | 单个 branch 超过 60 秒仍未认证；完整 840 branches 不可用 |
| root triangle/correlation cuts | 23,130 cuts 将 LP bound 从 `-1.1823723` 收紧到 `-0.9618250`；随后 MIP 超过 120 秒仍未认证 |

数值尺度也已修正：pricing objective 预乘 public `n`，直接在 global-gap 尺度
求解；HiGHS 使用 `1e-9` absolute/relative gap，并由代码再次验证 canonical
row objective和 dual bound。阻塞不再是 count/rate 缩放或 `mip_rel_gap=0`
造成的假慢。

按照专家冻结的 fallback 定义，`uncertified pricing` 属于
computational/certificate fallback。因此当前状态是：

```text
mechanism correctness: implemented
toy global theorem chain: passed
Adult Phase-I geometry: feasible with zero inflation
Adult exact pricing: computationally blocked
Stage 1 gate: not passed
formal utility panel: prohibited
AIM comparison: not run
```

这不是 C3 utility 的负结果，也不是 row polytope 与 confidence set 不相交；
它否定的是“在完整 105-pair shadow 上，以通用单线程 exact MILP 每轮获得
`1e-8` global pricing certificate”作为当前 paper-scale 子程序的可执行性。

## 18. 给专家的唯一待决问题

请在下列方向中冻结一个，不要同时授权多个。

### A. Bounded-treewidth shadow（当前工程首选）

公开冻结一个 bounded-treewidth unary/pairwise shadow graph，在该图上用 exact
variable elimination / junction-tree MAP 给出真正全局 pricing certificate；完整
released transcript仍供 final CCF-RCE-QDTE 使用。需明确 shadow edge selection、
maximum treewidth、未进入 shadow 的 blocks如何获得 pressure，以及 claim边界。

### B. Certified nonzero pricing interval

保留完整 105-pair shadow，但允许 solver返回 legal-row incumbent、global lower
bound和 nonzero FW-gap interval；只有 allocation对整个 interval稳健时才继续。
这需要重新冻结允许 gap、pressure robustification和 promotion gate。

### C. Separable/blockwise pressure

把 global pressure改为各 interaction block 的 exact local pricing，再由
water-filling汇总。它可高效且 exact per block，但不再是完整 row-polytope 的
global dual，需要新的数学定义。

### D. 停止 RHCG-CDWF paper branch

保留 Stage 0 和 Adult interior结果作为 diagnostic，回到已冻结的 QDTE/ICE
论文主线；把 full-row certified adaptive measurement留作后续 bounded-treewidth
或 low-rank工作。

在新决断前禁止：

```text
把未认证 incumbent 当 exact pricing
放宽后仍声称 G <= 1e-8
用 truth/evaluator 选 shadow edges
运行 formal seeds 或最终 utility panel
把 Stage 1 计算失败解释成 C3 utility 输给 AIM
```
