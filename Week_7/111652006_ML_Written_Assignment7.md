# Score Matching


在機率密度估計（density estimation）中，我們想學出一個模型分佈 $p_\theta(x)$，讓它接近真實資料分佈 $p(x; \theta)$。  
傳統的做法是maximum likelihood，但這需要計算難以求得的**分配函數（partition function）**：

$$
p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))
$$

這裡 $E_\theta(x)$ 是「能量函數」，而 $Z(\theta) = \int \exp(-E_\theta(x)) dx$ 很難計算。



## 定義與基本形式

假設模型分佈為：

$$
p(x; \theta) = \frac{1}{Z(\theta)} \exp(q(x; \theta))
$$

則其對數形式為：

$$
\log p(x; \theta) = q(x; \theta) - \log Z(\theta)
$$

定義 **score function** 為：

$$
S(x; \theta) = \nabla_x \log p(x; \theta) = \nabla_x q(x; \theta)
$$

score function 描述了在樣本空間中，機率密度上升最快的方向。

---

#  Score Matching Losses

##  Explicit Score Matching (ESM)

顯式分數匹配（Explicit Score Matching, ESM）直接最小化模型與真實分佈的 score 差距：

$$
L_{\text{ESM}}(\theta) = 
\mathbb{E}_{x \sim p(x)} \| S(x; \theta) - \nabla_x \log p(x) \|^2
$$

但在實際情況下，$\nabla_x \log p(x)$ 是未知的，因此 ESM 難以直接計算。

---

##  Implicit Score Matching (ISM)

隱式分數匹配（Implicit Score Matching, ISM）透過積分分部消去未知項，得到可實作的形式：

$$
L_{\text{ISM}}(\theta) = 
\mathbb{E}_{x \sim p(x)} \big[ 
\| S(x; \theta) \|^2 + 2 \nabla_x \cdot S(x; \theta)
\big]
$$

其中 $\nabla_x \cdot S(x; \theta)$ 表示 score function 的散度（divergence）。  
最小化 $L_{\text{ESM}}$ 與 $L_{\text{ISM}}$ 在理論上是等價的。

---

#  Denoising Score Matching (DSM)

##  DSM 的基本概念

**Denoising Score Matching (DSM)** 是對 ESM 的一種實際可行變體。  
我們先對資料加入雜訊，學習「有雜訊資料分佈」的 score function。

符號定義如下：

- $x_0$: 原始資料  
- $p_0(x_0)$: 原始資料分佈  
- $x$: 加入雜訊後的資料  
- $p(x|x_0)$: 有條件的（加雜訊）資料分佈  
- $p_\sigma(x)$: 加雜訊後的整體分佈  

目標是學習：

$$
S_\sigma(x; \theta) = \nabla_x \log p_\sigma(x)
$$

DSM 的損失函數定義為：

$$
L_{\text{DSM}}(\theta) =
\mathbb{E}_{x_0 \sim p_0(x_0)} 
\mathbb{E}_{x \sim p(x|x_0)} 
\| S_\sigma(x; \theta) - \nabla_x \log p(x|x_0) \|^2
$$

---

#  DSM Loss (Gaussian Noise 版本)

在實務中，我們通常加入高斯雜訊：

$$
x = x_0 + \epsilon, \quad
\epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

因此條件機率分佈為：

$$
p(x|x_0) = \frac{1}{(2\pi)^{d/2} \sigma^d} 
\exp \left( -\frac{1}{2\sigma^2} \| x - x_0 \|^2 \right)
$$

對此可得：

$$
\nabla_x \log p(x|x_0) 
= -\frac{1}{\sigma^2} (x - x_0)
= -\frac{1}{\sigma^2} \epsilon
$$

因此 DSM 的損失可改寫為：

$$
L_{\text{DSM}}(\theta) =
\mathbb{E}_{x_0 \sim p_0(x_0)}
\mathbb{E}_{x \sim p(x|x_0)}
\left\|
S_\sigma(x; \theta) + \frac{x - x_0}{\sigma^2}
\right\|^2
$$

或等價地表示為：

$$
L_{\text{DSM}}(\theta) =
\frac{1}{\sigma^2}
\mathbb{E}_{x_0, \epsilon}
\| \sigma S_\sigma(x_0 + \sigma \epsilon; \theta) + \epsilon \|^2
$$

---
#  Score-based Diffusion Models

在了解 Score Matching 與 Denoising Score Matching (DSM) 之後，  
我們可以將其應用到 **擴散式生成模型（Diffusion Generative Models）** 中。  
這類模型又常被稱為 **Score-based Generative Models (SGMs)**。

---

##  正向過程 (Forward Process)

在擴散模型中，我們對資料逐步加入高斯雜訊，使其最終趨近於標準高斯分佈。  
這個過程稱為**正向擴散（forward diffusion）**：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1 - \alpha_t) I)
$$

當 $t$ 越大，$x_t$ 越接近純雜訊。  
整個過程形成了一條「從真實資料到雜訊」的路徑。

---

##  反向過程 (Reverse Process)

接下來，我們希望從雜訊**逐步還原**資料。  
若已知每個時刻的真實 score function：

$$
\nabla_{x_t} \log p_t(x_t)
$$

理論上可以定義一個**反向隨機微分方程（Reverse SDE）**，  
讓我們能從雜訊樣本生成真實資料樣本：

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}
$$

其中：
- $f(x, t)$ 為擴散過程的漂移項（drift term）
- $g(t)$ 為擴散強度（diffusion coefficient）
- $d\bar{w}$ 為反向 Wiener 過程（reverse-time Brownian motion）

---

##  Score Network 的角色

由於真實的 $\nabla_x \log p_t(x)$ 無法取得，  
我們以神經網路 $s_\theta(x_t, t)$ 近似之：

$$
s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)
$$

在訓練時，我們利用 DSM loss（對應每個時間 $t$ 的噪音強度 $\sigma_t$）：

$$
L_{\text{DSM}}(\theta) =
\mathbb{E}_{t, x_0, \epsilon}
\| \sigma_t s_\theta(x_t, t) + \epsilon \|^2,
\quad x_t = x_0 + \sigma_t \epsilon
$$

這使得模型在各個噪音層級上學到「如何從雜訊中復原資料」。

---

##  生成過程 (Sampling)

訓練完成後，我們可以從標準高斯分佈取樣 $x_T \sim \mathcal{N}(0, I)$，  
並根據學到的 score function 解反向方程生成資料。  
常見的兩種生成方法如下：

### (1) Reverse SDE (Stochastic Sampling)
$$
dx = [f(x, t) - g(t)^2 s_\theta(x, t)] dt + g(t) d\bar{w}
$$

### (2) Probability Flow ODE (Deterministic Sampling)
$$
dx = [f(x, t) - \tfrac{1}{2} g(t)^2 s_\theta(x, t)] dt
$$

第二種方式對應 deterministic ODE，可用數值積分器（如 Runge–Kutta）生成樣本。  
實務上，這與 DDPM（Denoising Diffusion Probabilistic Models）的 sampling 過程等價。

---

##  總結比較表

| 類別 | 目標 | 損失形式 | 是否需真實 score | 應用 |
|------|------|-----------|------------------|------|
| ESM | 顯式匹配真實分佈的 score | $\mathbb{E}\|S - \nabla_x \log p\|^2$ |  需要 | 理論基礎 |
| ISM | 用散度消去未知項 | $\mathbb{E}[\|S\|^2 + 2\nabla_x \cdot S]$ |  不需要 | 實際可行 |
| Denoising Score Matching | 對加雜訊樣本進行訓練，讓模型學到「去雜訊方向」 |
| Score-based Diffusion Model | 透過訓練時間依賴的 score function，實現由雜訊生成資料 |

---

 **參考文獻：**
- Hyvärinen, Aapo. “Estimation of non-normalized statistical models by score matching.” *Journal of Machine Learning Research*, 2005.  
- Song, Yang & Ermon, Stefano. “Generative Modeling by Estimating Gradients of the Data Distribution.” *NeurIPS*, 2019.
  
