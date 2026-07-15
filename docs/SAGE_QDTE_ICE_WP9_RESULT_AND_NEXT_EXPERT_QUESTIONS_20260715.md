# SAGE-QDTE-ICE WP9 结果与下一步专家决策问题

Date: 2026-07-15

## 0. 希望专家拍板的问题

WP9 已经证明，当前完整 Static-ICE 不是一个跨 schema 稳定支配 AIM 的
统一默认方法，而是存在清晰的 cardinality regime：

```text
binary / low-cardinality:       ACS、NLTCS 显著优于 AIM
general-cardinality:            Adult、BR2000 稳定弱于 AIM
```

现在不缺更多 seed，也不缺更强 QDTE 搜索。需要决定下一步优先研究哪个
数学对象：

1. **改变 public allocation/planner 的风险目标**：从当前 cell-weighted
   total L2 variance，改成 block-normalized、TVD-risk 或公开
   cardinality-aware 的测量设计；
2. **引入 released sparse-tree prior**：从同一 DP transcript 通过
   post-processing 构造树结构先验，再由 QDTE 拟合完整 transcript；
3. 或者承认 ICE 是 binary/low-cardinality profile，不再追求当前论文中
   的统一低预算默认。

我们的初步判断是：**先请专家在 1 和 2 之间拍板，不继续调现有
`public_optimal` 公式，也不增加 QDTE 搜索强度。**

### 2026-07-15 WP9a public/released-only addendum

在不读取任何 true data/utility 的前提下，我们已经进一步审计了上述分叉。
发现当前 `public_optimal` 只对 complete one-/two-way **cell sub-workload**
精确最优，并没有对完整公开异构 workload 计算专家公式中的
`c_a=sum_q w_q ||L_qa||^2`。

精确 public factorization 显示：

```text
full-reconstructable-query-L2 optimal/current risk:
  ACS:     1.000000
  NLTCS:   1.000000
  BR2000:  1.000000
  Adult:   0.845572

equal-public-group-L2 optimal/current risk:
  ACS:     0.999653
  NLTCS:   0.999281
  BR2000:  0.655962
  Adult:   0.670850
```

同时，released tree 的两种 canonical dimension treatments 在 Adult/BR2000
上每个 seed 仅有约 `0.08--0.17` edge Jaccard，说明 tree prior 尚缺一个
非任意的维数校准定理。

因此本地优先级现修正为：**先完成 exact full-workload-L2 allocator 的
Adult causal pilot，再讨论 equal-group/TVD risk；tree prior 暂列其后。**
完整 no-truth 结果见：

```text
docs/SAGE_QDTE_ICE_WP9A_PUBLIC_ROUTE_DIAGNOSTIC_RESULT_20260715.md
```

## 1. WP9 冻结协议

Protocol:

```text
SAGE-QDTE-ICE-WP9-STATIC-CONFIRMATION-20260715-v1
```

Method:

```text
SAGE-QDTE-Static-ICE-Exact-v1
```

固定配置：

```text
adjacency:                 unbounded / add_remove
schema and n:              explicitly public
measurement:               all one-way Helmert contrasts
                           + all pair pure interactions under public cap
allocation:                public_optimal
projection:                frozen P1 path
precision:                 exact OrthogonalInteractionPrecision
QDTE iterations:           5000
candidates per iteration:  4096
candidates per cell:       20.48 million
transport:                 atom_flow
```

矩阵：

```text
datasets: NLTCS, ACS, BR2000, Adult
epsilon:  0.1, 0.3
seeds:    0, 1, 2
cells:    24
```

24 个 cell 全部在不读取 true utility 的情况下完成并封存。之后才运行
冻结 evaluator。所有 privacy、transcript hash、exact precision、迭代数、
candidate 数、incremental-state drift 和 no-truth gate 均通过。

证据：

```text
sealed panel:
  outputs/static_ice_wp9_formal_20260715/sealed_panel_manifest.json
  sha256 a61e918a398bc68912f927109e13a139d5d937568f3357431652df1463651ea0

frozen evaluation:
  outputs/static_ice_wp9_eval_20260715/gate_summary.json
```

## 2. 冻结 gate 结果

Primary ratio 定义为 Static-ICE / Official AIM；小于 1 表示 Static-ICE
更好。

### 2.1 总体与 dataset 结果

| Aggregate | Primary ratio |
| --- | ---: |
| Overall | **0.866260** |
| NLTCS | **0.708988** |
| ACS | **0.684284** |
| BR2000 | 1.064633 |
| Adult | 1.090229 |

总体几何平均好 13.4%，但这是由两个 binary dataset 的大胜覆盖了两个
general-cardinality dataset 的稳定退化，不能据此宣称 broad dominance。

### 2.2 Dataset-epsilon 结果

| Dataset | epsilon 0.1 | epsilon 0.3 |
| --- | ---: | ---: |
| NLTCS | **0.630006** | **0.797871** |
| ACS | **0.772984** | **0.605762** |
| BR2000 | 1.046122 | 1.083472 |
| Adult | 1.109228 | 1.071556 |

只有 `4/8` dataset-epsilon cells 获胜，而且恰好都是 binary datasets。

### 2.3 Seed 稳定性

BR2000 六个 seed-cell primary ratios 全部大于 1。Adult 六个中五个大于
1，仅 epsilon `0.3` seed `2` 为 `0.96185`。因此 general-cardinality
退化不是单 seed 偶然。

### 2.4 Bootstrap 与 tail

```text
hierarchical bootstrap 95%: [0.688382, 1.079772]
```

上界未低于 1。Adult dataset aggregate MaxTVD ratio 为 `1.234833`，超过
冻结的 `1.15` tail threshold。其余 dataset aggregate tail ratios没有触发
该阈值。

冻结 gate 因此失败：

```text
6/8 cell wins:                  fail (4/8)
overall primary < 1:            pass
bootstrap upper < 1:            fail
every dataset primary <= 1.05:  fail
every dataset tail <= 1.15:     fail

decision:
  retain_static_ice_as_development_candidate
```

## 3. 这不是 QDTE 搜索不足

每个 cell 都完整执行：

```text
5000 iterations
4096 candidates / iteration
20.48 million candidates scored
```

没有 confidence early stop，也没有 candidate shortfall。general-cardinality
的结果在三个 seed 上方向一致，所以继续扩大 candidate pool、增加迭代或
增加 seed 不是当前第一优先级。

epsilon `0.1` seed `0` 的 released standardized residual 如下：

| Dataset | Initial RMS | Final RMS |
| --- | ---: | ---: |
| NLTCS | 6.696 | 0.717 |
| ACS | 3.548 | 0.998 |
| BR2000 | 1.185 | 0.873 |
| Adult | 1.153 | 0.968 |

在 ACS/NLTCS，初始表离 transcript 有很强的可辨识信号，QDTE 能消除大部分
standardized residual。在 Adult/BR2000，初始表已经处于约一个 noise
standard deviation 内，QDTE 面对的 released signal 很弱。这个现象更像
测量统计效率与先验问题，而不是 row-edit optimizer 没有搜索够。

## 4. Cardinality 诊断

epsilon `0.1` 的公开 strategy 结构如下。所有数字仅由 public schema、
frozen strategy 和 privacy budget 得到。

| Dataset | Pair count | Pair coefficient dimension | Max pair dimension | Mean pair sensitivity | Pair rho share |
| --- | ---: | ---: | ---: | ---: | ---: |
| NLTCS | 120 | 120 | 1 | 0.500 | 64.5% |
| ACS | 253 | 253 | 1 | 0.500 | 69.2% |
| BR2000 | 91 | 3,446 | 300 | 0.682 | 84.3% |
| Adult | 105 | 13,165 | 600 | 0.842 | 92.0% |

binary pair 的 pure interaction 只有一个 coefficient，sensitivity 为
`0.5`。Adult pair 最多有 600 个 coefficients，sensitivity 接近 `1`。
因此“只测新增 interaction”在 binary schema 上有巨大的统计优势，在
general-cardinality schema 上优势显著减弱。

这不是当前 allocator 对 cell sub-workload 的实现 bug。代码严格实现：

\[
\rho_a \propto s_a\sqrt{c_a},
\]

其中当前 pair block 的

\[
c_a=(d_i-1)(d_j-1).
\]

它是专家文档闭式解在**所有重构 cell 等权总 L2 variance**上的特例。
WP9a 后已经确认：完整专家公式还应从实际公开 workload 计算 `c_a`，而
当前实现没有纳入 Adult 的 mixed/prefix/range 等可重构查询。

## 5. Target 与 generation 的已有证据

已有 offline measurement pilot 的 epsilon `0.1` seed `0` 结果：

| Dataset | ICE target RMSE | ICE target AvgPairTVD | QDTE final RMSE | QDTE final AvgTVD | AIM final RMSE | AIM final AvgTVD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NLTCS | 0.012529 | 0.020129 | 0.010064 | 0.016206 | 0.017710 | 0.030699 |
| ACS | 0.007796 | 0.012522 | 0.007469 | 0.009973 | 0.008041 | 0.011973 |
| BR2000 | 0.006892 | 0.104478 | 0.004174 | 0.056701 | 0.004437 | 0.051765 |
| Adult | 0.005759 | 0.188642 | 0.005202 | 0.080283 | 0.004952 | 0.074342 |

这张表说明：

1. binary datasets 的 released target 本身已经非常有竞争力；
2. Adult/BR2000 上 QDTE 的 row-realizability 会显著正则化 noisy target，
   但还不足以达到 AIM 的 block-distribution utility；
3. BR2000 的 RMSE 已能优于 AIM，但 AvgTVD 仍输，存在 block metric
   mismatch；
4. Adult 同时存在 measurement SNR 与结构先验差距。

## 6. 已经排除的直接补丁

以下机制均已在独立冻结 gate 中实现并验证，但没有通过 final utility
gate，不能在 WP9 后重新拼装救场：

```text
P3 + raw / active-set / BootDiag uncertainty
positive-part James-Stein interaction shrinkage
global confidence-constrained entropy with one-way product prior
interaction-aligned two-row cycles
normalized OI selector
selected-only private L1 endpoint
coverage-preserving private-L1 precision refinement WP8a
```

其中 Adult entropy arm 的 primary ratio 相对 exact raw 为 `1.35149`；
简单 product prior 太弱并导致 underfit。James-Stein shrinkage primary ratio
为 `1.73293` 或更差。不能把“再加一点 shrinkage/entropy”当成下一步。

完整 Static-ICE transcript 直接交给 Official Private-PGM 也不可行：Adult
all-pair graph 的 maximal clique 有 `424,688,379,494,400` cells，远超 20M
cap。

## 7. 请求专家回答的具体问题

### Q1. 当前 public risk objective 是否应该改变？

当前 `c_a` 对所有重构 cell 等权，因此高-cardinality pair 在总风险中占据
更大权重。若论文 primary 同时关心 query-level MAE/RMSE 和 equal-block
AvgTVD，是否应定义一个统一、public、可证明的 block-normalized risk，
例如令每个 pair marginal 的总权重相等，再推导新的

\[
c_a=\sum_q w_qL_{qa}^2,
\qquad
\rho_a\propto s_a\sqrt{c_a}?
\]

请给出推荐的 `w_q`、定理目标和 promotion protocol。我们不希望根据
Adult true errors手调 exponent 或 cardinality threshold。

### Q2. Public cardinality-aware scope filtering 是否有正当的 minimax/risk 依据？

如果 all-pair interaction 在 general-cardinality 下统计上不可取，能否只用
public schema 和 budget 定义 inclusion rule，例如基于预计 confidence radius
或 reconstruction risk，而不使用 true interaction strength？这样会牺牲
coverage；需要什么 sufficient condition 或 oracle inequality 才能避免变成
任意工程 heuristic？

### Q3. Released sparse-tree prior 是否比改 allocation 更值得优先？

一个候选路线是：

```text
full Static-ICE DP transcript
-> released-only pair score / public tie-break
-> sparse tree distribution p0
-> QDTE initialization or KL regularization
-> QDTE still scores the complete released transcript
```

它不额外访问 private truth，也避免 full all-pair PGM clique。请判断：

1. tree 应从 noisy interaction energy、mutual information 的 bias-corrected
   estimate，还是 confidence lower bound 选择？
2. tree prior 应只用于 initialization，还是进入
   `data objective + lambda KL(p || p0)`？
3. `lambda` 或 confidence radius 如何由 released covariance 和 public
   dimensions唯一确定，避免根据 Adult/BR2000 结果调参？
4. 能否给出 tree-prior QDTE 的统计风险解释或 oracle bound？

### Q4. 是否应接受 regime-specific ICE？

如果 Q1/Q3 都没有足够干净且可在当前论文范围内完成的数学方案，是否应
冻结为：

```text
QDTE-Base:                    broad default
QDTE-Static-ICE-Binary:       low-cardinality specialization
```

并把 general-cardinality confidence allocation / sparse prior 放到下一阶段？

## 8. 本地建议

在专家回复前：

1. 冻结 WP9，不增加 seed/epsilon，不改 gate；
2. 不再增强 QDTE 搜索；
3. 不对 Adult/BR2000 true error 做 scope 或权重调参；
4. 只做 released/public diagnostic，分别量化：
   - current cell-weighted risk；
   - equal-block/TVD upper-bound risk；
   - released tree 的 confidence stability；
5. 等专家在 Q1/Q3 之间拍板后，预声明一个 seed0 causal pilot。

WP9a 后本地优先级修正为：

```text
exact full-reconstructable-query-L2 allocator
>
有明确定理的 equal-group / TVD-related allocator
>
released sparse-tree prior
>
纯 cardinality threshold/filter
```

原因是当前 allocator 只完成了 total-cell L2 特例。完整公开 workload 在
Adult 上存在可证明的 `15.44%` measurement-risk 改善空间，而且不会改变
ACS/NLTCS/BR2000 的 exact full-query-L2 allocation。tree prior 仍可能补上
AIM 的结构先验，但当前 edge score 对维数归一化高度敏感，尚不适合先跑
utility 再选定义。
