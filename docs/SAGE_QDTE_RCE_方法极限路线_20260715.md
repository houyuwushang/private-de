---
title: "SAGE-QDTE-RCE：GCEA 之后继续追击 AIM 的方法极限路线"
subtitle: "Row-Realizable Confidence-Set Entropic Directed Evolution"
author: "专家研究决策稿"
date: "2026-07-15"
lang: zh-CN
documentclass: ctexart
classoption:
  - UTF8
geometry: margin=21mm
fontsize: 10.5pt
CJKmainfont: "Noto Serif CJK SC"
CJKsansfont: "Noto Sans CJK SC"
CJKmonofont: "Noto Sans Mono CJK SC"
mainfont: "Noto Serif CJK SC"
sansfont: "Noto Sans CJK SC"
monofont: "Noto Sans Mono CJK SC"
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - |
    \usepackage{amsmath,amssymb,bm,booktabs,longtable,array,mathtools}
    \usepackage{enumitem}
    \setlist{nosep,leftmargin=*}
    \usepackage{microtype}
    \setlength{\parindent}{2em}
    \setlength{\parskip}{0.22em}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[L]{\small SAGE-QDTE-RCE 方法极限路线}
    \fancyhead[R]{\small 2026-07-15}
    \fancyfoot[C]{\thepage}
    \setlength{\headheight}{15pt}
---

# 执行结论

## 拍板

我不接受“GCEA 没过 gate，所以 general-cardinality 到此为止”的结论。GCEA 只否定了一个更窄的命题：

> 在当前点目标、当前生成目标和当前统计估计器不变时，仅重新分配 measurement precision，不能稳定把 Adult 推过冻结 gate。

它没有否定 QDTE，也没有否定“全面追击 AIM”这个研究目标。

本稿拍板选择 **C 路线**：不再把下一步定义为另一个 allocator，而是把统计对象从“一个必须被拟合的 noisy point target”改成“包含真实 row distribution 的 Gaussian confidence set”，并让 QDTE 直接在 row-realizable empirical distributions 上寻找该置信集内的最大熵解。

```text
decision: C
method: SAGE-QDTE-RCE-v1
full name: Row-Realizable Confidence-Set Entropic QDTE
status: method-limit research track, implement now
not authorized as the next step:
  another static rho allocator
  another fixed entropy coefficient
  another post-hoc tail weight
  rejected-arm concatenation without a unified objective
```

核心变化只有一句话：

> **不再问“怎样更精确地拟合 noisy target”，而是问“在所有与 DP transcript 统计相容的 row-level tables 中，哪一张引入的额外结构最少”。**

这正面吸收 AIM 在低预算下最重要的统计优势——不对噪声支持不足的依赖关系做过度解释——同时保留 QDTE 相对 Private-PGM 更灵活的 row-level optimization、异构 workload 支持和精确 edit scoring。

## 为什么是 C，而不是继续 B 式分配

当前 binary/general-cardinality 分界与 pure-interaction sensitivity 完全一致。对属性域大小 $d_j,d_k$，纯 pair interaction 的 add/remove $L_2$ sensitivity 为

$$
\Delta_{jk}^{\mathrm{int}}
=
\sqrt{1-\frac1{d_j}}
\sqrt{1-\frac1{d_k}}.
$$

binary $\times$ binary 时它等于 $1/2$；基数增大时趋近 1。也就是说，ICE 在 binary 数据上天然获得强方差优势，而一般基数数据上这一优势减弱。GCEA 又表明，改变平均 measurement risk 可以改善 MAE/RMSE，却仍可能把风险集中到少量坐标并伤害 MaxError。因此，下一步需要改变的是 **估计器的偏差-方差行为和 row-realizable geometry**，而不是继续优化同一个点估计目标的预算权重。

AIM 的公开实现本身也明确假设数据已经处理到没有 large-cardinality categorical attributes，并建议在存在大基数属性时使用 MST 的 domain compression；它每轮用 L1-minus-noise 选择 marginal，再通过 MirrorDescent/Private-PGM 重估分布，并根据模型变化与预期噪声比较进行退火。这里真正难打的是统计模型和正则化，不只是 allocator。

# 1. 当前失败究竟否定了什么

WP10a/GCEA 的因果信息应分成三层。

第一，公开 risk 公式和实现可以完全正确，但 evaluator transfer 仍失败。measurement risk、projected target risk、row-realizable fit 和 final evaluator utility 是不同对象。

第二，Adult 的 MAE/RMSE 同向小幅改善，说明一般基数不是“完全没有可利用信号”，而是现有方法把信息转成最终 row table 的方式不够统计稳健。

第三，Structured QDTE 在 clean same-target 条件下可以打赢强 GSD，但在 noisy target 上更容易把 released loss 降低而不改善真实 utility。这说明 generator capacity 不是缺失项；缺失项是 **只拟合具有统计证据的结构**。

因此，正确研究问题是：

$$
\boxed{
\text{怎样让 QDTE 只实现 transcript 能够显著支持的依赖结构，
并对最终 row-level output 给出可验证的误差包络？}
}
$$

# 2. RCE 的数学对象

## 2.1 记号

设原始离散域为 $\mathcal X$，真实数据的 empirical distribution 为

$$
p_D\in\mathcal P_n
=
\left\{
\frac{c}{n}:
 c\in\mathbb Z_+^{|\mathcal X|},\ \mathbf1^\top c=n
\right\}.
$$

公开 measurement operator 为 $A$。DP transcript 在 rate space 中写成

$$
z=Ap_D+\xi,
\qquad
\xi\sim\mathcal N(0,\Sigma).
$$

$\Sigma$ 可以是 diagonal、block、low-rank 或由 exact precision operator 隐式表示。若 consistency/post-processing 使协方差奇异，则所有范数和 rank 均限制在可识别 support 上，并使用 Moore--Penrose inverse。

## 2.2 同时控制平均误差与 pointwise tail 的置信集

固定总失败概率

$$
\alpha=0.05,
\qquad
\alpha_2=0.025,
\qquad
\alpha_\infty=0.025.
$$

令 $r=\operatorname{rank}(\Sigma)$，$m$ 为进入 release objective 的坐标数，定义

$$
c_2=F^{-1}_{\chi^2_r}(1-\alpha_2),
$$

$$
c_\infty
=
\Phi^{-1}\left(1-\frac{\alpha_\infty}{2m}\right).
$$

定义 mixed confidence set：

$$
\mathcal C(z)
=
\left\{
 p\in\mathcal P_n:
 (Ap-z)^\top\Sigma^\dagger(Ap-z)\le c_2,
 \quad
 \max_q
 \frac{|(Ap-z)_q|}{\sqrt{\Sigma_{qq}}}
 \le c_\infty
\right\}.
\tag{1}
$$

第一个约束控制 aggregate whitened error；第二个约束阻止总风险通过牺牲少数 coordinates 来下降。它们分别对应 WP10a 暴露出的 average-risk 和 MaxError 两个问题。

对于多个独立或条件独立 block，也可以逐 block 使用 $c_{2,g}$，并以预声明的 Bonferroni allocation $\alpha_{2,g}=\alpha_2/G$ 保持同样保证。第一版建议同时保留 global ellipsoid 和 coordinatewise tube，不引入 family-specific 权重。

## 2.3 Released one-way maximum-entropy prior

从已发布并投影到 simplex 的 one-way counts $\widetilde a_j$ 构造严格正的 public/released prior：

$$
\bar p_j(c)
=
\frac{\widetilde a_j(c)+1}{n+d_j},
$$

$$
p_0(x)=\prod_{j=1}^d\bar p_j(x_j).
\tag{2}
$$

这里的 pseudocount 固定为 1，不以 dataset、truth 或 evaluator 调参。$p_0$ 保留 released one-way 信息；在没有足够 interaction 证据时，它回退到近独立、最大熵的结构。

## 2.4 唯一 estimator

RCE estimator 定义为

$$
\boxed{
\widehat p
=
\arg\min_{p\in\mathcal C(z)}
D_{\mathrm{KL}}(p\|p_0).
}
\tag{RCE}
$$

它有非常清楚的统计含义：

- 若 $p_0$ 已与 transcript 统计相容，则不凭噪声制造 interaction；
- 若某些 released residual 超出 confidence region，则只引入使输出重新进入置信集所必需的结构；
- 在所有统计上可接受的 row-level tables 中，选择相对 released one-way prior 额外信息最少的一张。

这不是固定 $\lambda$ 的“quadratic loss + entropy penalty”。固定 penalty coefficient 会再次引入调参和尺度问题。RCE 使用 constraint form，regularization strength 由 Gaussian noise、rank 和预声明置信水平自动决定。

## 2.5 永远可运行的 lexicographic 版本

数值算法可能暂时找不到可行 integer table。禁止在看到结果后放宽 confidence radius。使用唯一的 lexicographic fallback：

第一阶段最小化最大标准化违反量

$$
s^\star
=
\min_{p\in\mathcal P_n}
\max\left\{
\frac{(Ap-z)^\top\Sigma^\dagger(Ap-z)}{c_2}-1,
\max_q\frac{|(Ap-z)_q|}{c_\infty\sqrt{\Sigma_{qq}}}-1,
0
\right\}.
$$

第二阶段在所有达到 $s^\star$ 的 tables 中最小化 $D_{\mathrm{KL}}(p\|p_0)$。若 $s^\star=0$，就是原始 RCE；若 $s^\star>0$，它给出一个明确的 optimizer/measurement feasibility certificate，而不是静默改阈值。

# 3. 可以写进论文的理论链

## 3.1 隐私定理

若 $z$ 由 $\rho$-zCDP measurement mechanism 产生，而 $p_0$、$\mathcal C(z)$、candidate generation、dual update、QDTE scoring、transport、stopping 和最终 table 全部只依赖 $z$、公开 schema、公开 workload 和随机数，则 RCE-QDTE 输出仍满足 $\rho$-zCDP。

若后续依据已发布 transcript 决定追加 Gaussian measurements，则每轮 selection 是 post-processing；privacy ledger 只累加实际 Gaussian $\rho$，并满足 adaptive zCDP composition。

## 3.2 高概率可行性

在 noise/covariance specification 正确时，

$$
\Pr[p_D\in\mathcal C(z)]\ge1-\alpha.
$$

因此真实 empirical distribution 自身以至少 $0.95$ 概率证明 integer feasible set 非空。这一点非常重要：RCE 不是把 query-space target 投影到一个可能与 row tables 不相交的集合，而是直接在包含真实 row table 的随机置信集合中求解。

## 3.3 final row-level utility 包络

在事件 $p_D,\widehat p\in\mathcal C(z)$ 上，由三角不等式：

$$
\left\|
\Sigma^{\dagger/2}A(\widehat p-p_D)
\right\|_2
\le 2\sqrt{c_2}.
\tag{3}
$$

对每个进入 coordinatewise tube 的 query：

$$
\boxed{
|(A\widehat p-Ap_D)_q|
\le
2c_\infty\sqrt{\Sigma_{qq}}.
}
\tag{4}
$$

因此这是一个 **final synthetic table** 的 MaxError guarantee，而不是 WP10a/GCEA 那种 measurement-only risk envelope。

对于 complete partition block $b$：

$$
\operatorname{TVD}_b(\widehat p,p_D)
\le
\frac12\sum_{q\in b}
|(A\widehat p-Ap_D)_q|
\le
c_\infty\sum_{q\in b}\sqrt{\Sigma_{qq}}.
\tag{5}
$$

若该 block 有独立、等方差 coordinates，还可用 $L_2$ 约束得到

$$
\operatorname{TVD}_b
\le
\frac12\sqrt{d_b}\,
\|A_b(\widehat p-p_D)\|_2.
\tag{6}
$$

保证只覆盖进入 transcript/confidence operator 的 evaluator directions；未测且不可线性重构的 directions 不能被虚构成已有 theorem。它们应通过原有 heterogeneous measurements、额外 public sketch 或独立报告处理。

## 3.4 relaxed estimator 的唯一性

把 $\mathcal P_n$ 放松到 simplex $\Delta(\mathcal X)$ 后，confidence set 是 convex，且当 $p_0(x)>0$ 时 $D_{\mathrm{KL}}(p\|p_0)$ 严格凸。因此 relaxed RCE 有唯一解。

这给出两个有用对象：

1. relaxed optimum 是统计 estimator 的上限；
2. integer QDTE 与 relaxed optimum 的 gap 是纯 row-realization/optimizer gap。

这正好把“估计器不够好”和“QDTE 没吃下 estimator”分开。

## 3.5 KL row edit 的精确有限差分

设 synthetic table 中完整 row atom $x$ 的 count 为 $c_x$，$p_x=c_x/n$。对一次 $u\to v$ replacement，KL 项的精确变化为

$$
\begin{aligned}
\Delta_{\mathrm{KL}}(u\to v)
=\frac1n\Bigg[&
(c_u-1)\log\frac{c_u-1}{np_0(u)}
-c_u\log\frac{c_u}{np_0(u)}\\
&+(c_v+1)\log\frac{c_v+1}{np_0(v)}
-c_v\log\frac{c_v}{np_0(v)}
\Bigg],
\end{aligned}
\tag{7}
$$

约定 $0\log0=0$。因此 entropy/max-entropy regularization 不破坏 QDTE 的 exact-scoring 主线。

## 3.6 primal-dual QDTE

RCE 的 Lagrangian 可写为

$$
\mathcal L(p,\lambda,\mu^+,\mu^-)
=
D_{\mathrm{KL}}(p\|p_0)
+
\lambda\left[(Ap-z)^\top\Sigma^\dagger(Ap-z)-c_2\right]
+
\sum_q\mu_q^+[(Ap-z)_q-b_q]
+
\sum_q\mu_q^-[-(Ap-z)_q-b_q],
$$

其中

$$
b_q=c_\infty\sqrt{\Sigma_{qq}},
\qquad
\lambda,\mu^+,\mu^-\ge0.
$$

固定 dual variables 时，一次 row edit 的 Lagrangian change 等于：

1. 已有 general-PSD quadratic exact advantage；
2. coordinate tube 的线性 exact term；
3. 式 (7) 的 exact KL term。

因此可以沿用 QDTE candidate compiler、GPU scoring 和 conflict-aware transport。dual variables 由约束违反量更新；最终必须报告 primal feasibility、dual feasibility、complementarity 和 primal-dual gap。

# 4. 为什么它有机会在一般基数上反超 AIM

## 4.1 它不是再拟合一个 noisy point

当前 point-target QDTE 的诱惑是：只要 released loss 还能下降，就继续实现 residual。高基数 block 中有大量 noise directions，强 generator 会更容易过拟合。

RCE 把目标改成 set-valued：进入 confidence region 后，继续降低 point residual 不再有收益；优化器转而最大化熵。它把“Base 的隐式正则”变成显式、可证明且与噪声尺度一致的正则。

## 4.2 维度越高，自动 shrink 得越强

高维 Gaussian block 的 $\chi^2$ radius 随有效 rank 增长。若 released interaction 只相当于 noise，one-way product prior 往往已经在 confidence set 内，RCE 不会制造细粒度相关；只有超出统计阈值的 aggregate interaction 才会激活 dual pressure。

这正是 general-cardinality 缺少的机制。它不是手工按 $d_b$ 加权，而是由 exact noise covariance 和 effective rank 决定需要解释多少结构。

## 4.3 binary 优势不会被主动抹掉

binary pure interaction 的 sensitivity 和 coefficient variance更低，confidence region相对更窄；真实 interaction 更容易形成显著 constraint violation，QDTE会继续实现这些结构。换言之，同一机制会在 binary regime 更接近当前 ICE full fit，在 high-cardinality/low-SNR regime 更接近 maximum-entropy shrinkage。

## 4.4 它吸收 PGM 的强项，但不接受 PGM 的表示上限

AIM 每轮根据 noisy measurements 重估一个图模型，这是它低预算稳定性的主要来源；公开代码还用 junction-tree size filter 限制可加入的 cliques。RCE 同样使用 maximum-entropy principle，但优化变量是 row-realizable empirical distribution，不要求最终结果受一个固定 graphical-model clique structure 限制。

因此其理想分工是：

```text
AIM-like strength:
  noise-aware maximum-entropy inductive bias

QDTE-specific strength:
  unrestricted legal rows
  heterogeneous query effects
  exact row-edit finite differences
  structured cycles
  direct row-level output
```

# 5. general-cardinality 的第二层：支持自适应粗化

RCE-core 应先在现有冻结 transcript 上运行，以单独检验 estimator。若 Adult/BR2000 的 relaxed RCE ceiling 明显高于 current ICE，第二层才启用 **released one-way support coarsening**。它不是另一个自由 variant，而是 RCE 的计算/统计实现。

设所有属性总类别数为

$$
D_{\mathrm{tot}}=\sum_jd_j,
$$

冻结

$$
\beta_{\mathrm{sup}}=0.01,
\qquad
\kappa_{\mathrm{sup}}
=
\Phi^{-1}\left(1-\frac{\beta_{\mathrm{sup}}}{2D_{\mathrm{tot}}}\right).
$$

对已发布 one-way count $z_{j,c}$ 和标准差 $\sigma_{j,c}$，定义

$$
H_j
=
\{c:z_{j,c}-\kappa_{\mathrm{sup}}\sigma_{j,c}>0\},
$$

$$
R_j=\mathcal X_j\setminus H_j.
\tag{8}
$$

$H_j$ 中类别保留 singleton；$R_j$ 合并为一个 rare bucket。若 $R_j=\varnothing$，映射为 identity。若 $H_j=\varnothing$，保留 released count 最大的类别为 singleton，其余合并，以避免退化实现。

之后的 pair/interaction vector 直接在 coarse domain 上测量，conditional sensitivity 仍为 1。该 partition 只依赖先前 DP one-way release，因此无需额外 selection privacy charge；追加 Gaussian measurements照常进入 zCDP ledger。

最终 QDTE 仍在原始类别空间生成 rows。coarse measurements 只约束 aggregated cells，$p_0$ 中的 released one-way probabilities决定 rare bucket内部的默认分配。因此不需要 MST 式 uniform decompression，也不需要接触 truth。

这层机制的数学作用是：

- 减少 interaction block 的 effective rank；
- 避免把预算用于 one-way noise 下无法确认存在的类别；
- 对 binary/低基数高频属性通常保持 identity；
- 对 rare-category-specific interactions明确承认无法恢复，并由 maximum entropy填充，而不是拟合噪声。

# 6. Gaussian fission：探索方法极限时的 final-risk 选择器

RCE 的 confidence set已经给出 final-output envelope，但如果要比较一条固定的 QDTE checkpoint path，例如 product prior、首次可行点、minimum-KL point、进一步 objective refinement，就需要一个不使用 truth 的选择器。

对任一 Gaussian release

$$
z=a+\xi,
\qquad
\xi\sim\mathcal N(0,\Sigma),
$$

再生成独立

$$
\eta\sim\mathcal N(0,\Sigma).
$$

固定 $\tau=1/2$，定义

$$
z_{\mathrm{tr}}=z+\tau\eta,
$$

$$
z_{\mathrm{val}}=z-\tau^{-1}\eta.
\tag{9}
$$

则

$$
\operatorname{Cov}(z_{\mathrm{tr}}-a,z_{\mathrm{val}}-a)
=
\Sigma-\Sigma=0.
$$

由于二者联合 Gaussian，它们独立，且

$$
\Sigma_{\mathrm{tr}}=1.25\Sigma,
\qquad
\Sigma_{\mathrm{val}}=5\Sigma.
$$

这是对已发布 DP transcript 的随机 post-processing，不增加 privacy cost；代价是把 Fisher information 按 80/20 分给 train/validation。

## 6.1 paired final-risk theorem

所有 candidate tables $s_0,\ldots,s_K$ 只由 $z_{\mathrm{tr}}$ 生成。对任意 public PSD evaluator matrix $M$，定义相对 baseline $s_0$ 的风险差估计：

$$
\widehat\Delta_{k,M}
=
\|s_k-z_{\mathrm{val}}\|_M^2
-
\|s_0-z_{\mathrm{val}}\|_M^2.
\tag{10}
$$

条件于 candidate tables：

$$
\mathbb E[\widehat\Delta_{k,M}]
=
\|s_k-a\|_M^2-
\|s_0-a\|_M^2,
$$

且

$$
\operatorname{Var}(\widehat\Delta_{k,M})
=
4(s_k-s_0)^\top
M\Sigma_{\mathrm{val}}M
(s_k-s_0).
\tag{11}
$$

因此可用 simultaneous Gaussian upper confidence bound，只在 candidate 对预声明的 RMSE risk 和 TVD-$L_2$ upper-bound risk 都被证明显著更好时替换 baseline。MaxError 则使用

$$
U_{\infty}(s_k)
=
\max_q
\left[
|s_{k,q}-z_{\mathrm{val},q}|
+c_{\mathrm{val}}\sqrt{(\Sigma_{\mathrm{val}})_{qq}}
\right].
\tag{12}
$$

这比根据 true top-error query 设计 guard 更干净：它保护的是 final synthetic output，而且选择器只读独立 DP validation view。

第一版不应把 fission 和 RCE-core 同时作为唯一因果变化。正确顺序是先证明 RCE point-to-set estimator本身有 ceiling，再加 fission 解决 checkpoint/stopping。

# 7. 不浅尝辄止的极限实验程序

这里不采用“一个 seed0 没过就停止”的研究方式。整个 C 路线按 **统计 ceiling、row-realization ceiling、DP mechanism、broad validation** 四层推进。每一层失败都指向下一类数学问题。

## 7.1 C0：四格 ceiling decomposition

```text
datasets: Adult, BR2000
epsilon: 0.1, 0.3
seeds: 0,1,2
generator budget: frozen 5000 x 4096
```

固定 evaluator 和 QDTE，实现四个 diagnostic cells：

| Partition / support | Interaction information | 回答的问题 |
|---|---|---|
| released | DP transcript | 完整可发布机制 |
| oracle | DP transcript | support/coarsening 是否是瓶颈 |
| released | clean interaction target | measurement noise 是否是瓶颈 |
| oracle | clean interaction target | estimator + generator 的能力上限 |

oracle cells 只用于因果诊断，绝不作为 paper method，也不能把 oracle 选择反向编码进最终规则。

同时计算 relaxed RCE optimum 和 integer QDTE-RCE：

```text
AIM vs relaxed RCE:
  statistical/model-class ceiling

relaxed RCE vs integer QDTE-RCE:
  row-realization / optimizer gap

point-target QDTE vs RCE-QDTE:
  noisy-target overfit gap
```

这一步不是 gate 后“放弃”，而是决定下一步究竟应该强化 measurement、prior 还是 optimizer。

## 7.2 C1：same-transcript RCE 因果实验

使用 WP9/WP10a 已封存的完全相同 transcript：

```text
control: current Static-ICE point-target QDTE
candidate: RCE-QDTE-core
same z / Sigma / init / RNG / candidate pool / iterations
only changed object:
  point target -> row-realizable confidence set + minimum KL
```

必须报告：

```text
minimum confidence slack s*
primal feasibility
coordinate tube violation
KL to released one-way prior
relaxed optimum
integer gap
active dual blocks
accepted single edits / cycles
final true metrics, offline only
```

C1 至少完成 Adult/BR2000 两预算三 seeds，不以 seed0 决策。

## 7.3 C2：support-adaptive direct coarse measurement

只有 C1 表明 RCE estimator 有收益但 high-card interaction noise仍是主瓶颈时，启用式 (8) 的唯一 support rule，并重新直接测量 coarse complete partitions。不能先在 full noisy table上聚合后冒充 direct coarse measurement，因为前者会累计多个 cell noises，失去真正的维数收益。

对照必须包括：

```text
same support partition + direct coarse measurement
same support partition + aggregate full measurement
```

两者之差就是 direct coarse measurement 的真实统计收益。

## 7.4 C3：confidence-dual adaptive refinement

当 RCE 已工作后，才重开 adaptive measurement。下一 measurement action不再由 generic L1 或旧 SAGE harmonic score决定，而由 RCE dual certificate决定。

对 block $g$，设当前 dual multiplier 为 $\lambda_g$，其 public confidence width 在追加 $\Delta\rho$ 后从 $w_g(\rho_g)$ 降为 $w_g(\rho_g+\Delta\rho)$。定义 released-only value of information：

$$
\operatorname{VOI}_g
=
\lambda_g
\left[
 w_g(\rho_g)-w_g(\rho_g+\Delta\rho)
\right].
\tag{13}
$$

选择 VOI 最大且满足 public resource cap 的 block。它的含义是：把下一份预算投给当前 RCE 解最受其 uncertainty 限制的约束，而不是投给 raw residual 最大的 block。selection 只读 released transcript 和 dual variables，因此不花 selection privacy；Gaussian refinement按实际 $\rho$ 记账。

# 8. 一次 broad promotion 不是研究终止条件

最终 broad evaluation：

```text
datasets: Adult, BR2000, ACS, NLTCS
epsilon: 0.1, 0.3, 1, 3, 10
seeds: 0-4 for 0.1/0.3
       0-2 for 1/3/10
baselines:
  official AIM, official settings
  current Static-ICE
  GCEA frozen diagnostic
  RCE-QDTE
  RCE-QDTE + support coarsening, only if C2 admitted
```

研究目标仍然是 broad dominance，但结果解释按以下层级：

| 结果模式 | 数学解释 | 下一动作 |
|---|---|---|
| relaxed RCE 胜 AIM，integer RCE 输 | generator/oracle gap | 强化 cycle compiler、column generation、transport |
| oracle support 胜，released support 输 | support estimator不足 | spectral/low-rank category embedding，而不是再调 allocator |
| clean interaction 胜，DP interaction 输 | measurement information不足 | C3 dual-driven refinement或low-rank sketch |
| RCE primary 胜、tail输 | coordinate tube/solver未落实 | 修 certificate；不得用 true tail选位置 |
| relaxed 与 integer 都输 AIM | product prior/model class不足 | 升级为 released tree或low-rank interaction prior |
| binary 回退、general改善 | confidence calibration过度保守 | 检查 covariance/rank与solver，不按 dataset调阈值 |

这张表意味着：一次失败不会自动推出“做不到”。只有当 **oracle support + clean interactions + relaxed RCE** 仍系统性输 AIM，才说明当前“confidence-set maximum entropy”模型类本身没有足够 ceiling；届时应转向 low-rank/latent interaction prior，而不是回到 allocator sweep。

# 9. 代码实现拍板

建议新增：

```text
qdte/rce/confidence_set.py
  global chi-square ellipsoid
  coordinatewise Gaussian tube
  singular covariance support

qdte/rce/prior.py
  released one-way Dirichlet(1) product prior

qdte/rce/dual.py
  primal-dual state
  feasibility / complementarity certificate

qdte/evolution/rce_scoring.py
  PSD quadratic exact term
  linear tube term
  exact KL row-edit term

qdte/measurement/support_coarsening.py
  simultaneous-LCB support rule
  original-to-coarse maps

qdte/validation/gaussian_fission.py
  train/validation views
  covariance audit

qdte/validation/paired_risk.py
  equations (10)--(12)
```

必须新增测试：

```text
true empirical table lies in simulated confidence set at nominal coverage
relaxed RCE uniqueness on small domains
KL single-edit exact difference vs full recomputation
quadratic + linear + KL batch exactness
singular covariance / zero-precision support
primal-dual KKT residuals
binary identity support map
high-card rare-bucket sensitivity == 1
fission train/validation empirical covariance == 0
paired risk estimator unbiasedness
final MaxError envelope coverage
no truth object reachable from generation process
```

# 10. 与论文现有工作量的关系

这条路线不要求删除你们已有理论。相反，它把原本分散的结果连成一条更强主链：

1. complete/interaction measurement sensitivity给出 $\Sigma$；
2. P3 的 convex geometry成为 query-space confidence set的前身；
3. projection-to-generation counterexample解释为什么必须直接优化 row distribution；
4. QDTE exact edit theorem成为 RCE integer solver；
5. structured cycles成为保持低阶 marginals时实现显著 interactions 的原子；
6. transfer decomposition被升级为 final row-output confidence theorem；
7. GCEA negative result证明 average-risk allocation不足以替代 pointwise/row-level geometry。

因此 RCE 不是另起炉灶，也不是把 rejected arms重新拼起来。它是把同一个核心问题改写成正确的数学对象。

# 11. 最终决断

```text
choose: C
freeze name: SAGE-QDTE-RCE-v1
first implementation: same-transcript RCE-core
first scientific target:
  distinguish estimator ceiling from integer QDTE ceiling

then:
  support-adaptive coarse measurement
  confidence-dual adaptive refinement
  fission final-risk selection

not a stopping rule:
  one failed seed
  one failed allocator
  one failed fixed-lambda entropy arm
```

最值得坚持的一句话是：

$$
\boxed{
\text{AIM 的低预算优势不是不可逾越的“PGM 魔法”，
而是它没有把每个 noisy residual都当成必须实现的事实。}
}
$$

SAGE-QDTE-RCE 用一个更一般的方式获得同样的统计克制：真实 table以高概率位于 confidence set 中；QDTE不再追逐置信集内部的噪声差异，而是在 row-realizable tables 中寻找额外结构最少的解。随后，只有被 confidence dual证明真正受 uncertainty 限制的方向才获得更多 measurement precision。

这是当前最有可能同时保住 binary 优势、修复 general-cardinality、控制 MaxError，并把“最终 row-level utility”纳入 theorem 的路线。它不能在实验前被诚实地称为已经全面打赢 AIM；但它已经把“为什么可能打赢”从直觉提升为一个可实现、可证伪、可逐层定位上限的数学程序。

# 参考材料

1. Ryan McKenna, Brett Mullins, Daniel Sheldon, Gerome Miklau. *AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data*. 2022.
2. Ryan McKenna et al. `mbi/mechanisms/aim.py` and `mst.py`, official public implementation.
3. Samuel Maddock, Shripad Gade, Graham Cormode, Will Bullock. *GEM+: Scalable State-of-the-Art Private Synthetic Data with Generator Networks*. 2025.
4. 项目文档：*SAGE-QDTE 项目全貌、当前矛盾与专家集中决策问题*, 2026-07-13.
5. 项目研究设计：*SAGE-QDTE-ICE：面向低隐私预算、以击败 AIM 为目标的完整方法升级方案*, 2026-07-13.
6. 项目诊断：*QDTE Projection-to-Generation Transfer Gap*, 2026-07-08.
