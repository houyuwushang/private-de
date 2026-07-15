# SAGE-QDTE Adult Post-GCEA Expert Decision Request

Date: 2026-07-15

## 0. 我们现在请求的不是另一个 rescue variant

专家此前为 Q1-B 预声明了唯一的 general-cardinality allocator：
`SAGE-QDTE-Static-ICE-GCEA-v1`。我们已严格执行公开机制门；四个
Adult/BR2000、epsilon 0.1/0.3 cells 均未达到 `t_star <= 0.97`，因此：

```text
mechanism_gate_passed: false
generation_authorized: false
decision: q1_a_freeze_ice_binary_profile
```

我们接受这个结果，不要求放宽阈值、重跑 GCEA、删除 tail guard，也不要求
把 GCEA 与 P3、entropy、cycles、selector 或 sparse tree 临时拼接。

现在用户仍希望知道一个更高层的问题：

> 在不进行 post-hoc truth tuning、并把当前论文冻结决策与后续研究分开的
> 前提下，Adult 低预算场景是否仍存在一个数学上值得继续、最终可能超过
> Official AIM 的新方法方向？

请专家判断的是“是否开启一个独立 post-GCEA research track”，而不是为
当前失败的 gate 找例外。

可复核的 WP10b 快照：

```text
https://github.com/anonymous-owner/private-de/tree/exp/sage-qdte-ice-wp10b-gcea-20260715
```

## 1. 当前 Adult 差距到底有多大

WP9 的 frozen end-to-end protocol 使用 unbounded/add-remove DP、Static-ICE、
P1、exact interaction precision 和 QDTE-Standard。每个 cell 均运行 5000
iterations、每轮 4096 candidates，共 20.48M candidate scores。

Primary composite 是 MAE、RMSE、AvgTVD 三项 ratio 的几何平均，ratio 定义为
Static-ICE / Official AIM，小于 1 表示我们更好：

| Dataset | epsilon 0.1 | epsilon 0.3 | aggregate |
| --- | ---: | ---: | ---: |
| NLTCS | 0.630006 | 0.797871 | 0.708988 |
| ACS | 0.772984 | 0.605762 | 0.684284 |
| BR2000 | 1.046122 | 1.083472 | 1.064633 |
| Adult | **1.109228** | **1.071556** | **1.090229** |

Adult 六个 seed-epsilon cells 中五个输，仅 epsilon 0.3 seed 2 为 0.96185。
Adult aggregate MaxTVD ratio 为 1.234833，也超过冻结的 1.15 tail threshold。
这不是单 seed 偶然，也不是 generator 没跑满。

epsilon 0.1 seed 0 的代表性指标：

| Method | RMSE | AvgTVD |
| --- | ---: | ---: |
| Static-ICE target | 0.005759 | 0.188642 pair-TVD |
| QDTE final | 0.005202 | 0.080283 |
| Official AIM final | 0.004952 | 0.074342 |

QDTE 已经把 noisy target 显著正则化到 row-realizable table，但仍未跨过 AIM。
因此差距约为 5% RMSE 和 8% AvgTVD 量级，不是数量级差距，却跨 seed 稳定。

## 2. 为什么继续增加 QDTE 搜索量不是答案

Adult epsilon 0.1 的 released standardized residual RMS：

```text
initial: 1.153
final:   0.968
```

而 NLTCS/ACS 初始分别为 6.696/3.548，final 为 0.717/0.998。Adult 初始表
已经在约一个 noise standard deviation 内，可辨识的 residual-directed signal
很弱。继续增大 candidate pool 或 iterations 更可能拟合 transcript noise，
而不是补上统计信息。

Adult 的公开 strategy 几何也不同：

```text
pair scopes:                 105
pair coefficient dimension: 13,165
max pair dimension:          600
mean pair sensitivity:       0.842
pair rho share:              about 92%
```

binary pair 只有一个 pure interaction coefficient 且 sensitivity 为 0.5；
Adult 的 interaction dimension 大、sensitivity 接近 1。ICE 的 binary 优势在
这里自然衰减。

## 3. allocation-only 路线已经得到什么反证

### WP10a: full reconstructable-query L2 allocation

Adult 28,654 evaluator queries 中，22,534 个可由当前 strategy 精确重构；
6,120 个高阶 queries 不在闭式 allocation objective 中。

公开 measurement L2 risk 改善 15.44%，但 end-to-end seed0 结果为：

| Metric | Candidate / Control |
| --- | ---: |
| MAE | 0.974694 |
| RMSE | 0.971309 |
| AvgTVD | 1.015487 |
| MaxTVD | 1.003192 |
| MaxError | 1.359274 |

说明平均 L2 allocation 有真实收益，但与 block-L1 和 pointwise tail 不一致。

### WP10b: GCEA evaluator envelopes

GCEA 用公开 MAE/RMSE/AvgTVD measurement risks 做 minimax，并把公开 Gaussian
MaxError/MaxTVD envelopes 硬约束为不差于 control。公开结果：

| Dataset | epsilon | t_star | MaxError envelope | MaxTVD envelope |
| --- | ---: | ---: | ---: | ---: |
| Adult | 0.1 | 0.994034 | 0.977083 | 0.946160 |
| Adult | 0.3 | 0.994034 | 0.977083 | 0.946161 |
| BR2000 | 0.1 | 1.000000 | 1.000000 | 1.000000 |
| BR2000 | 0.3 | 1.000000 | 1.000000 | 1.000000 |

Adult 在同时保护三个 primary surrogates 与两个 tail envelopes 后只剩约 0.6%
的最坏 primary 改善，远低于预声明的 3%。这强烈暗示：在**当前固定
measurement strategy**下，仅改变 rho allocation 已接近收益边界。

WP10b 有 KKT/cross-epsilon numerical certificate failures，因此我们不把
`t_star=1` 写成精确全局不可能性定理；但四个 cells 的 0.97 gate 独立失败，
足以维持冻结决策。

## 4. 已经跑过且不能原样再推荐的路线

以下机制均已有独立冻结实验或诊断，不能把它们原样组合后称为新方法：

```text
P3 + raw / active-set / BootDiag uncertainty
positive-part James-Stein interaction shrinkage
global confidence-constrained entropy with one-way product prior
interaction-aligned two-row cycles
normalized OI selector
selected-only private partition-L1 endpoint
coverage-preserving private-L1 precision refinement
full-reconstructable-query L2 allocation
GCEA evaluator-envelope allocation
larger QDTE candidate pool / more iterations
```

Adult entropy arm相对 exact raw 的 primary ratio 为 1.35149，简单 one-way
product prior 明显 underfit。James-Stein shrinkage ratio 为 1.73293 或更差。
不能再建议“加一点 entropy/shrinkage”。

Official Private-PGM 在 sparse partial transcript 上对 Adult 比 QDTE 好约
9%--15%，说明 maximum-entropy inductive bias确实有价值；但它相对 full
Static-ICE 和 AIM 仍很差。把 full all-pair Static-ICE transcript直接交给
Private-PGM 也不可行，因为 Adult all-pair graph 的 maximal clique 约有
`4.2469e14` cells，远超资源上限。

## 5. 请专家先判断：固定 strategy 是否已经走到边界

### Q1. allocation-only 是否可以正式关闭？

请判断以下结论是否成立：

> WP10a 和 WP10b 已足以说明，在当前 one-way + all-pair pure-interaction
> strategy、当前 evaluator 和 tail safety要求下，继续改变 public rho
> allocation 不太可能稳定弥合 Adult 对 AIM 的约 7%--11% end-to-end gap。

如果不同意，请指出 GCEA 数学目标、重构定义或实现中的**具体错误**。仅仅
不喜欢 `t_star <= 0.97` 或 tail约束，不构成重开理由。

### Q2. 能否给出 fixed-strategy 的公开 lower bound 或 Pareto certificate？

是否可以从 public strategy matrix、sensitivity、rho 和 evaluator reconstruction
map 推导一个 lower bound / Pareto frontier，回答：

```text
任意非自适应 linear Gaussian measurement strategy，
在同时控制 query-average L2、block-L1 和 coordinate tail 时，
Adult 当前 basis 至少要承担多少 measurement risk？
```

如果能够证明 fixed strategy 的统计余量不足，后续研究就应明确转向 strategy
或 inductive bias，而不是继续调 allocator。

## 6. 如果仍有机会，真正不同的新机制应是什么

请不要同时授权多个 arm。请先比较以下机制类别，再选择**至多一个**。

### Candidate A: general-cardinality strategy redesign

当前 strategy 对每个 pair 测完整 `(d_i-1)(d_j-1)` interaction basis。是否应
改成一个有 approximation-risk bound 的 multiresolution / low-rank / workload-
factorized strategy，只测 evaluator真正需要且低噪的 interaction directions？

需要专家给出：

```text
exact public strategy construction
sensitivity theorem
approximation + noise risk bound
对 6,120 unsupported high-order queries 的处理
为什么不是看 Adult truth 后做 basis truncation
```

### Candidate B: sparse confidence graph + broad QDTE

是否可以从 full DP transcript 和 covariance 构造一个 confidence-certified
sparse graph/tree prior，同时保留 broad Static transcript作为 QDTE objective：

```text
released interaction confidence lower bounds
-> unique sparse graph/tree rule
-> max-entropy initialization or constrained prior
-> QDTE fits the complete released transcript
```

这需要一个不依赖 true MI、true errors 或 observed winner 的 threshold，并说明
prior strength如何由 covariance/public dimensions唯一确定。请解释它为何能
避开此前 one-way product prior underfit 和 full-PGM clique explosion。

### Candidate C: confidence-set inference instead of penalty tuning

此前 entropy 失败使用的 prior/penalty可能过强。是否存在一个无自由 lambda 的
约束形式：

\[
\min_p D_{\mathrm{KL}}(p\|p_0)
\quad\text{s.t.}\quad
(Ap-y)^\top\Sigma^\dagger(Ap-y)\le \tau,
\]

其中 `tau` 由公开自由度和预声明 coverage probability唯一确定，并由 QDTE
exact edits近似求解？如果推荐，必须说明它与已失败的 global entropy arm 在
数学和实现上有何本质区别，以及为什么不会再次 underfit。

### Candidate D: low-round private adaptive sparse measurement

是否必须承认 AIM 的优势来自 private adaptive clique selection，而非 Static
strategy可以弥合？若是，请给出一个 4--8 round、count-scale、敏感度明确的
机制及完整 zCDP split，并解释它与已失败 normalized OI selector 和 WP8a
private-L1 refinement的本质区别。

## 7. 学术上如何允许继续，而不变成追着 Adult 调参

Adult、ACS 及现有 seeds 已被反复查看。若专家认为仍值得继续，请同时给出
validation firewall。我们的建议下限是：

```text
Adult existing seeds/results:
  development and mechanism diagnosis only

new mechanism:
  exact formula and all hyperparameters frozen before new outcomes

promotion evidence:
  new untouched seeds and preferably a new general-cardinality dataset
  paired noise and paired initialization
  full evaluator, not selected favorable queries
  Official AIM unchanged
  all failed and successful cells reported
```

请判断：一个在 Adult 开发后只在新的 Adult seeds上验证是否足够，还是必须
增加至少一个此前未用于 method selection 的 general-cardinality dataset。

## 8. 希望专家按以下格式最终拍板

请只选择一个主决策：

```text
Decision A:
  当前 paper 和后续短期研究都停止 Adult rescue。
  接受 ICE 的 binary/low-cardinality boundary。

Decision B:
  当前 paper 仍冻结，但授权一个独立 post-GCEA research track。
  从 Candidate A/B/C/D 中只选一个，并给出完整 mechanism、theorem 和
  holdout protocol。成功后再决定是否形成下一篇或独立扩展。

Decision C:
  发现 WP10b/GCEA 存在会改变结论的明确数学或实现错误。
  请指出错误、正确公式和不接触 true utility 的修复审计；在审计完成前
  不授权 generation。
```

若选择 B，请务必补全：

```text
method name:
single changed component:
public/released/private inputs:
privacy accounting:
objective:
theorem or certificate:
fixed hyperparameters:
resource cap:
development data:
untouched promotion data:
primary gate:
tail gate:
failure fallback:
whether eligible for current paper or follow-up only:
```

## 9. 本地当前判断

我们的当前判断是：

1. **当前论文不应再救 Adult。** Q1-A fallback 应继续有效。
2. **allocation-only 可以关闭。** WP10a/WP10b 已经给出足够强的负证据。
3. Adult 仍非数学上不可战胜；约 7%--11% gap 和 partial-PGM diagnostic说明
   可能存在新方法空间，但它更可能需要 strategy redesign 或可证的 sparse
   structural prior，而不是更强 QDTE 搜索。
4. 若专家无法给出唯一机制和 holdout firewall，应选择 Decision A。

我们请求专家大胆判断“是否还有真正的新科学问题”，而不是为了得到一次
Adult 胜利而继续堆实验。
