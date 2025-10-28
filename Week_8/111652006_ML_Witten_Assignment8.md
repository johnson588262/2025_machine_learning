# Written assignment

## (1) Sliced Score Matching (SSM)

我們要證明 **Sliced Score Matching (SSM)** 也可以寫成如下:

$$
L_{\text{SSM}}(\theta)
= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}
\Big[\|v^{\top}S(x;\theta)\|^2 + 2\,v^{\top}\nabla_x(v^{\top}S(x;\theta))\Big].
$$

---

## Proof

根據我們知道的定義, SSM loss 如下:

$$
L_{\text{SSM}}(\theta)= \mathbb{E}_{x\sim p(x)}\|S(x;\theta)\|^2+ \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}\Big[2v^{\top}\nabla_x(v^{\top}S(x;\theta))\Big].
$$

所以，我們也可以說題目要我們證明的就是:

$$
\mathbb{E}_{x\sim p(x)}\|S(x;\theta)\|^2
= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}\|v^{\top}S(x;\theta)\|^2.
$$

---

對於任意 fixed $x$ (so $S := S(x;\theta) \in \mathbb{R}^d$ is fixed), 我們有:

$$
\mathbb{E}_{v\sim p(v)}\|v^{\top}S\|^2
= \mathbb{E}_{v\sim p(v)}\big[(v^{\top}S)^{\top}(v^{\top}S)\big]
= \mathbb{E}_{v\sim p(v)}\big[S^{\top}(vv^{\top})S\big].
$$

因為 $S$ 是 fixed 的，且 respect to $v$, 所以我們可以把它移出期望值:

$$
\mathbb{E}_{v\sim p(v)}\|v^{\top}S\|^2
= S^{\top}\mathbb{E}_{v\sim p(v)}[vv^{\top}]S.
$$

令 $v$ 是一個 isotropic random vector 並且滿足:

$$
\mathbb{E}_{v\sim p(v)}[vv^{\top}] = I,
$$

where $I$ is the identity matrix.  
(This holds if $v$ is sampled uniformly on the unit sphere or from $\mathcal{N}(0,I)$.)

所以

$$
\mathbb{E}_{v\sim p(v)}\|v^{\top}S\|^2
= S^{\top}IS = \|S\|^2.
$$

---
等式兩邊都給一個 $\mathbb{E}_{x\sim p(x)}$:

$$
\mathbb{E}_{x\sim p(x)}\|S(x;\theta)\|^2
= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}\|v^{\top}S(x;\theta)\|^2.
$$

所以

$$
L_{\text{SSM}}(\theta)
= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}
\Big[\|v^{\top}S(x;\theta)\|^2 + 2v^{\top}\nabla_x(v^{\top}S(x;\theta))\Big]. \quad\square
$$

## (2) Briefly explain SDE

**Stochastic Differential Equation（SDE，隨機微分方程）** 描述含噪動態的連續時間系統：

$$
dx_t = f(x_t,t)\,dt + G(x_t,t)\,d W_t,
$$

其中 $f$ 為漂移（drift），$g$ 為擴散係數（diffusion），$w_t$ 是 Wiener 過程（布朗運動）。
描述了一個系統在deterministic dynamics與andom noise共同作用下的演化過程。
它可以被視為常微分方程（ODE） 的隨機版本，亦即在經典動態系統中加入隨機擾動後的對應形式。

在 **score-based / diffusion** 生成模型中：
- 正向過程用 SDE 把資料逐步加噪，趨近高斯；
- 反向過程可由 **reverse-time SDE** 生成資料： $dx_t = \big[f(x_t,t) - g(x_t,t)^2\,\nabla_x \log p_t(x)\big]dt + g(x_t,t)\,d\bar{W}_t,$ 以學得的 score $S(x,t)\approx\nabla_x\log p_t(x)$ 取代未知真值即可取樣；  
- 亦可用對應的 **probability flow ODE** $dx = \big[f(x_t,t) - \tfrac{1}{2}g(x_t,t)^2\,S(x,t)\big]dt$ 進行確定性（無噪）生成。
