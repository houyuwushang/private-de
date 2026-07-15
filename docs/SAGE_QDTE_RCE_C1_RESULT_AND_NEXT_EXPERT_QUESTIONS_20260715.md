# SAGE-QDTE RCE C1 结果与下一轮专家决策问题

日期：2026-07-15  
状态：C1 已完成盲生成、整体封存和一次冻结离线评估；本文档只用于研究诊断，不反向修改 C1。

## 0. 希望专家拍板什么

请不要再给出可自由拼接的组件清单。请基于下面的新证据，回答两个层次的问题：

1. **Adult 是否仍值得继续进攻 Official AIM？**
2. 若值得，下一步只允许哪一个主机制：
   - `C0` ceiling decomposition；
   - `C3` confidence-dual adaptive refinement；
   - released tree / low-rank interaction prior；
   - 一个明确的 row-atom/cycle/column-generation oracle；
   - 或停止 Adult rescue，正式承认方法边界。

如果选择继续，请给出唯一、可预声明、不能读取 true utility 调参的完整数学定义和实验 gate。

## 1. 背景与当前对手差距

冻结的 WP9 `Static-ICE` 在 Adult 低预算上尚未打赢 Official AIM，但差距原本不是数量级差距：

| Adult | WP9 / AIM primary composite |
|---|---:|
| epsilon 0.1 | 1.109228 |
| epsilon 0.3 | 1.071556 |
| aggregate | 1.090229 |

因此我们按照专家给出的 RCE 路线，检验下面的机制：

```text
point-target QDTE
  -> row-realizable Gaussian confidence set
  -> minimum KL to released one-way product prior
  -> exact row-edit primal-dual QDTE
```

RCE 希望在低预算时只实现有统计证据支持的 interaction，避免强生成器拟合 Gaussian 噪声。

## 2. 冻结 C1 协议

```text
protocol: SAGE-QDTE-RCE-C1-20260715-v3
datasets: Adult, BR2000
epsilon: 0.1, 0.3
seeds: 0, 1, 2
generator: 5000 iterations x 4096 candidates
adjacency: unbounded / add_remove
```

每个 candidate 与 WP9 control 共用：

```text
同一个 sealed DP transcript
同一个 schema 和 public n
同一个 initial table
同一个 generation seed
同一个 candidate budget
同一个 orthogonal interaction precision
同一个 evaluator
```

唯一变化是生成目标：

```text
WP9: exact point-target weighted quadratic
RCE: confidence feasibility first, then minimum D_KL(p || p0)
```

其中：

```text
p0 = released complete one-way product prior, pseudocount 1
alpha_l2 = 0.025
alpha_linf = 0.025
RCE regularizer = D_KL(p_empirical || p0), rate-space scale
```

正式 v3 之前发现并修复了两个问题：

1. v1 错把 `n * D_KL` 与 O(1) dual update 混用；
2. v2 搜索尺度已正确，但 JSON certificate 又将 normalized KL 除以 `n`。

v1/v2 都在看 true utility 前停止，零个 cell 被封存。v3 增加了代码哈希、KL scale、certificate 和 transcript-only hard gate。

## 3. 盲评结论

### 3.1 Primary composite

所有 ratio 均为 RCE / 对照，越小越好。

| Dataset | epsilon | RCE / WP9 | RCE / AIM |
|---|---:|---:|---:|
| Adult | 0.1 | 1.711710 | 1.898677 |
| Adult | 0.3 | 1.027912 | 1.101466 |
| BR2000 | 0.1 | 1.491170 | 1.559946 |
| BR2000 | 0.3 | 1.269192 | 1.375134 |

聚合：

| Slice | RCE / WP9 | RCE / AIM |
|---|---:|---:|
| Adult | 1.326457 | 1.446142 |
| BR2000 | 1.375711 | 1.464628 |
| Overall | 1.350860 | 1.455356 |

结论非常明确：**RCE-v1 没有改善冻结 point-target control，也没有打赢 AIM。**

Adult `epsilon=0.3, seed2` 的 primary ratio 对 AIM 为 `0.996835`，但其他 Adult seeds 都输，不能把单 seed 持平解释成成功。

### 3.2 Adult 分指标

| epsilon | denominator | MAE | RMSE | AvgTVD | MaxTVD | MaxError |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | WP9 | 1.7811 | 1.9522 | 1.4424 | 0.8344 | 1.5967 |
| 0.1 | AIM | 2.0189 | 2.1343 | 1.5885 | 1.2114 | 1.2502 |
| 0.3 | WP9 | 1.0212 | 1.0627 | 1.0008 | 1.0378 | 1.1096 |
| 0.3 | AIM | 1.1474 | 1.0904 | 1.0681 | 1.0900 | 0.6903 |

RCE 在部分 MaxTVD/MaxError 上有收益，但 primary average utility 明显不足，尤其 `epsilon=0.1`。

## 4. 置信集合、KL 和 solver 证据

在完整面板封存后，我们运行了单独的 offline post-seal diagnostic。它没有参与生成或调参。

### 4.1 Truth coverage

| Arm | inside confidence set |
|---|---:|
| true empirical table | 12 / 12 |
| shared initial table | 0 / 12 |
| WP9 point-target final | 8 / 12 |
| RCE-v1 final | 9 / 12 |

真实表 `12/12` 均在集合内。这支持当前 Gaussian covariance/rank/confidence construction 的 nominal coverage，并说明 Adult `epsilon=0.1` 的正 slack 不是“真实可行集本身为空”。

### 4.2 Adult feasibility pattern

| epsilon | seed | RCE slack | ellipsoid ratio | tube ratio |
|---:|---:|---:|---:|---:|
| 0.1 | 0 | 0.047660 | 1.047660 | 1.001064 |
| 0.1 | 1 | 0.052821 | 1.052821 | 0.852455 |
| 0.1 | 2 | 0.047549 | 1.047549 | 1.001236 |
| 0.3 | 0 | 0 | 0.845416 | 0.872143 |
| 0.3 | 1 | 0 | 0.854542 | 0.783478 |
| 0.3 | 2 | 0 | 0.847532 | 0.934110 |

Adult `epsilon=0.1` 三个 seed 都留下约 5% 的 global ellipsoid gap；`epsilon=0.3` 三个 seed 全部整数可行。BR2000 六个 cells 全部可行。

### 4.3 Product-prior KL

跨 12 cells 平均：

| Arm | mean D_KL to released product prior |
|---|---:|
| initial | 6.828177 |
| RCE-v1 | 5.305168 |
| truth | 7.161322 |
| WP9 control | 7.671770 |

按关键 slice：

| Slice | RCE KL | Truth KL | WP9 KL |
|---|---:|---:|---:|
| Adult 0.1 | 6.169200 | 9.634134 | 9.781022 |
| Adult 0.3 | 7.336390 | 8.518718 | 9.246987 |
| BR2000 0.1 | 3.503026 | 5.386106 | 5.991287 |
| BR2000 0.3 | 4.212058 | 5.106330 | 5.667785 |

RCE 的确显著降低了相对 product prior 的 KL，但真实数据和高效用 WP9 table 都需要更多 interaction information。现有 RCE 似乎把“统计相容”解释得过于保守。

### 4.4 RCE 不是完全没有生成收益

Primary composite：

```text
RCE / initial: 0.653627
WP9 / initial: 0.483860
```

RCE 相比 released-one-way initial table 改善约 34.6%，说明 exact RCE row edits 确实实现了部分结构；但 point-target WP9 改善约 51.6%，保留了更多有效 interaction。

更关键的反例是 BR2000：WP9 在 `6/6` cells 中也位于 RCE confidence set 内，并且 true utility 显著更好；RCE 选择了更低 KL、但更差的表。因此 **只修 integer feasibility 或继续增加候选，不足以解释或解决全部失败**。

### 4.5 Released-only restricted-mixture 诊断

为了进一步区分 integer row-edit trajectory 与 estimator objective，我们在不读取
truth 的独立进程中冻结并求解了下面的三表凸包：

```text
p(w) = w_initial p_initial + w_wp9 p_wp9 + w_rce p_rce
w >= 0, sum(w) = 1
```

它保留 C1 的同一个 confidence operator 和 released one-way product prior，先最小化
confidence slack，再在最优 slack face 上最小化完整 row-distribution
`D_KL(p(w) || p0)`。KL 在三张表的 full-row atom union 上计算，不是把三张表当成
三个类别计算。全部 12 个解先封存，随后才运行冻结的 full-workload evaluator。

该对象只是在声明的三表凸包上的 restricted relaxed diagnostic，**不是全局 relaxed
RCE ceiling**。

平均 component weights：

| initial | WP9 control | RCE-v1 |
|---:|---:|---:|
| 0.079089 | 0.092829 | 0.828083 |

Primary composite：

| Slice | Restricted / RCE-v1 | Restricted / WP9 | Restricted / AIM |
|---|---:|---:|---:|
| Adult epsilon 0.1 | 0.831426 | 1.423161 | 1.578610 |
| Adult epsilon 0.3 | 0.967185 | 0.994182 | 1.065321 |
| BR2000 epsilon 0.1 | 0.879869 | 1.312035 | 1.372549 |
| BR2000 epsilon 0.3 | 0.916134 | 1.162750 | 1.259807 |
| Overall | 0.897279 | 1.212098 | 1.305861 |

这个结果补上了一个重要缺口：

1. fractional interpolation 相对 integer RCE 平均改善约 `10.3%`，所以存在次要的
   integer/search-path gap；
2. 但 constrained minimum-KL 解仍平均给 RCE `82.8%` 权重，只给高 utility WP9
   `9.3%` 权重；
3. 它仍比 WP9 差 `21.2%`，比 AIM 差 `30.6%`；
4. Adult `epsilon=0.1` 的 fractional mixture 已将 integer feasibility gap 消到数值
   tolerance，但仍比 WP9 差 `42.3%`。

因此，在这个明确包含 WP9 方向的 convex relaxation 中，去掉 integer interpolation
障碍并没有让 minimum-product-KL 选择高 utility 方向。它比原来的相关性证据更直接地
支持：**当前主要失败来自 product-prior minimum-KL estimator geometry；integer QDTE
不是第一瓶颈。**

## 5. 我们当前的判断

1. **RCE-v1 本身已经被否证为 broad replacement。** 不应继续调 alpha、KL scale 或根据 true metric 选 checkpoint。
2. **当前证据已经把 one-way product prior / minimum-KL estimator 过度收缩定位为第一瓶颈。** 它把真实 interaction 与 noisy interaction 一起丢掉了。
3. integer/search-path gap真实存在但属于次要项；restricted fractional mixture改善 RCE，却仍主动避开 WP9 方向并明显输给 WP9。
4. **Adult 仍不构成数学不可能。** 冻结 WP9 对 AIM 原本只差约 7%--11%；但下一步若继续，更合理的是引入唯一、DP-safe 的 released structural prior，而不是继续优化当前 product-prior RCE 或直接假定 measurement 不足。

## 6. 请专家回答的集中问题

### Q1：这个证据是否已经足以判定 product-prior RCE 目标错误？

加入 restricted-mixture 结果后，请明确判断下面哪一项最符合证据：

```text
A. 证据已足以把 product-prior minimum-KL 判为第一瓶颈；
B. 仍主要是 frozen transcript 信息不足；
C. 仍主要是 integer/global row oracle 不足；
D. A 与 B 都成立，但请说明同一 transcript 下 WP9 可行反例为何仍不足以优先修 prior。
```

尤其请解释：BR2000 的 WP9 table 已在集合内且 utility 更好，而 RCE 以更低 KL 选择更差 table，这是否已经是 minimum-product-KL estimator 不对齐的充分诊断？

### Q2：Adult 还值得继续吗？

请在以下两项中拍板：

```text
STOP:
  当前论文停止 Adult rescue，保留 WP9/QDTE broad method，
  将 product-prior RCE 写成 negative diagnostic。

CONTINUE:
  Adult 原 WP9 距 AIM 只有 7%--11%，仍值得做一个新机制。
  但只能指定一个主机制和一次预声明实验。
```

### Q3：如果继续，下一个唯一机制是什么？

基于 restricted-mixture 结果，我们认为优先级已经不再对称：

```text
1. preferred: richer released prior:
   released tree / low-rank / latent interaction prior

2. only with a new justification: tighter information:
   C3 confidence-dual adaptive refinement or direct low-rank sketch
```

请单选，并给出：

```text
完整 objective
prior/measurement 如何只依赖 public 或 DP-released information
总 rho 如何固定和记账
是否沿用同一 confidence set
row edit 的 exact finite-difference 公式
预声明的 rounds/rank/tree construction
禁止调节的参数
datasets/epsilon/seeds
对 WP9 和 AIM 的 promotion gate
失败后的停止条件
```

### Q4：如果仍要求先做 C0，请把三个未定义对象写死

原路线中的 C0 目前还不能无歧义实现。请明确：

1. `oracle support`：是真实非零 one-way categories、真实 top-k categories、真实最优 coarse partition，还是其他唯一规则？
2. `clean interaction target`：是把 noisy coefficient mean 替换为 truth、但保留原始 Sigma 和 confidence radius；还是设为 zero variance 的 exact target？两者回答的问题不同。
3. Adult/BR2000 巨大 row domain 上的 `relaxed RCE ceiling`：atom support 如何构造？用什么全局 MAP/column oracle或 reduced-cost certificate，才能避免把 restricted optimum误称为 global ceiling？

如果 C0 只是 restricted diagnostic，请给出何种结果足以授权下一机制，何种结果必须停止。

### Q5：是否需要一个介于 point fit 与 minimum-KL 之间、但不含自由 lambda 的对象？

当前两个极端是：

```text
WP9 point fit:      utility 较好，但可能拟合噪声；
RCE minimum KL:     统计上保守，但明显丢失有效 interaction。
```

若专家认为应使用 constrained information projection、minimum excess-risk、structural prior confidence set或其他对象，请直接给出唯一公式。不要建议事后调 `lambda`，除非 lambda 能由公开 confidence certificate 唯一确定。

## 7. 证据路径

```text
frozen protocol:
  docs/SAGE_QDTE_RCE_C0_C1_PROTOCOL_20260715.md

sealed blind panel:
  outputs/sage_qdte_rce_c1_v3_20260715/sealed_panel_manifest.json

frozen offline evaluation:
  outputs/sage_qdte_rce_c1_v3_eval_20260715/summary.json
  outputs/sage_qdte_rce_c1_v3_eval_20260715/metrics.csv

post-seal confidence/KL diagnostic:
  outputs/sage_qdte_rce_c1_v3_postseal_20260715/summary.json
  outputs/sage_qdte_rce_c1_v3_postseal_20260715/confidence_and_kl.csv

released-only restricted-mixture seal:
  outputs/sage_qdte_rce_c1_restricted_mixture_v2_cpu_20260715/sealed_manifest.json

frozen restricted-mixture offline evaluation:
  outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/summary.json
  outputs/sage_qdte_rce_c1_restricted_mixture_eval_v2_20260715/weights.csv
```

## 8. 一句话问题

> RCE-v1 以及包含 WP9 方向的 restricted convex-hull optimum 都显示，one-way product-prior minimum-KL 会主动压低有效 interaction；integer interpolation 不是第一瓶颈。Adult 原始 WP9 仍只差 AIM 约 7%--11%。请判断是否值得用一个唯一的 DP-released structural prior 继续解决；若否则冻结为方法边界，若继续则给出不可自由调节的完整公式与一次性 gate。
