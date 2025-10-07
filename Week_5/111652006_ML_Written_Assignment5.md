# Assignment5
## 問題一：多變量常態分布的歸一化

給定：

$$
f(x) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

其中：
- $x, \mu \in \mathbb{R}^k$
- $\Sigma$ 是 $k\times k$ 的正定矩陣（positive definite matrix）
- $|\Sigma|$ 是其行列式（determinant）

證明：

$$
\int_{\mathbb{R}^k} f(x)\, dx = 1
$$

---

令：

$$
y = \Sigma^{-1/2}(x - \mu) \quad \Rightarrow \quad x = \Sigma^{1/2}y + \mu
$$

此變換會讓指數中的橢球形二次項變成標準形式。

Jacobian為：

$$
dx = |\det(\Sigma^{1/2})|\, dy = |\Sigma|^{1/2}\, dy
$$

代入積分項:

$$
\begin{aligned}
\int_{\mathbb{R}^k} f(x)\, dx
&= \frac{1}{\sqrt{(2\pi)^k|\Sigma|}}
   \int_{\mathbb{R}^k}
   e^{-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)}\, dx \\
&= \frac{1}{\sqrt{(2\pi)^k|\Sigma|}}
   \int_{\mathbb{R}^k}
   e^{-\frac{1}{2}y^Ty}\, |\Sigma|^{1/2} dy
\end{aligned}
$$

因為

$$
\frac{1}{\sqrt{|\Sigma|}} \times |\Sigma|^{1/2} = 1
$$

因此：

$$
\int_{\mathbb{R}^k} f(x)\, dx
= \frac{1}{(2\pi)^{k/2}}
   \int_{\mathbb{R}^k} e^{-\frac{1}{2}y^Ty}\, dy
$$

而後者是標準 $k$-維常態分布的積分結果，為 1。

$$
\boxed{
\int_{\mathbb{R}^k} f(x)\, dx = 1
}
$$


##  證明：k 維標準常態分布的積分等於 1

我們要證明

$$
\int_{\mathbb{R}^k} \frac{1}{(2\pi)^{k/2}} e^{-\frac{1}{2}\|y\|^2}\,dy = 1
$$

因為

$$
e^{-\frac{1}{2}\|y\|^2}
= e^{-\frac{1}{2}(y_1^2 + y_2^2 + \cdots + y_k^2)}
= \prod_{i=1}^k e^{-\frac{1}{2}y_i^2},
$$

所以由 Tonelli 或 Fubini 定理（因為函數非負），可以分離成：

$$
\int_{\mathbb{R}^k} e^{-\frac{1}{2}\|y\|^2}\,dy
= \prod_{i=1}^k \int_{\mathbb{R}} e^{-\frac{1}{2}y_i^2}\,dy_i.
$$

令

$$
I = \int_{\mathbb{R}} e^{-\frac{1}{2}x^2}\,dx,
$$

則上式變為

$$
\int_{\mathbb{R}^k} e^{-\frac{1}{2}\|y\|^2}\,dy = I^k.
$$


考慮平方：

$$
I^2 = \left( \int_{\mathbb{R}} e^{-\frac{1}{2}x^2}\,dx \right)
       \left( \int_{\mathbb{R}} e^{-\frac{1}{2}y^2}\,dy \right)
     = \iint_{\mathbb{R}^2} e^{-\frac{1}{2}(x^2 + y^2)}\,dx\,dy.
$$

改用極座標：

$$
x = r\cos\theta, \quad y = r\sin\theta, \quad dx\,dy = r\,dr\,d\theta.
$$

則：

$$
I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-\frac{1}{2}r^2}\,r\,dr\,d\theta
    = 2\pi \int_0^{\infty} e^{-\frac{1}{2}r^2}\,r\,dr.
$$

代換變數 \(u = \frac{1}{2}r^2 \Rightarrow du = r\,dr\)，得

$$
I^2 = 2\pi \int_0^{\infty} e^{-u}\,du = 2\pi.
$$

所以

$$
I = \sqrt{2\pi}.
$$

$$
\int_{\mathbb{R}^k} e^{-\frac{1}{2}\|y\|^2}\,dy = I^k = (\sqrt{2\pi})^k = (2\pi)^{k/2}.
$$

因此

$$
\int_{\mathbb{R}^k} \frac{1}{(2\pi)^{k/2}} e^{-\frac{1}{2}\|y\|^2}\,dy
= \frac{(2\pi)^{k/2}}{(2\pi)^{k/2}} = 1.
$$

總而言之

$$
\boxed{
\int_{\mathbb{R}^k} \frac{1}{(2\pi)^{k/2}} e^{-\frac{1}{2}\|y\|^2}\,dy = 1
}
$$

此即證明 k 維標準常態分布（mean = 0, covariance = I）的總機率為 1。

---

## 問題二：矩陣微分與跡（Trace）性質

### (a)

$$
\frac{\partial}{\partial A} \mathrm{trace}(AB) = B^T
$$

因為

$$
\mathrm{trace}(AB) = \sum_{ij} A_{ij}B_{ji}
\Rightarrow \frac{\partial}{\partial A_{ij}} \mathrm{trace}(AB) = B_{ji}
$$

因此：

$$
\frac{\partial}{\partial A} \mathrm{trace}(AB) = B^T
$$

---

### (b)

$$
x^T A x = \mathrm{trace}(x x^T A)
$$

因為

$$
x^T A x \in 1 * 1\text{ matrices}
$$

所以

$$
x^T A x = \mathrm{trace}(x^T A x)
$$

然後trace裡面可以在不製造錯誤矩陣乘法的情況下移動位置，所以

$$
x^T A x = \mathrm{trace}(x^T A x) = \mathrm{trace}(x x^T A)
$$

---

### (c)：多變量常態分布的最大似然估計（MLE）

對樣本 $x_1, \ldots, x_N \sim \mathcal{N}(\mu, \Sigma)$，其對數似然為：

$$
\log L(\mu, \Sigma)
= -\frac{N}{2}\log|2\pi\Sigma|
  -\frac{1}{2}\sum_{i=1}^N (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)
$$

---

對 $\mu$ 求導：

$$
\frac{\partial \log L}{\partial \mu}
= \Sigma^{-1}\sum_i (x_i - \mu) = 0
\Rightarrow
\boxed{
\hat{\mu} = \frac{1}{N}\sum_i x_i
}
$$

---

對 $\Sigma$ 求導：

$$
\frac{\partial \log L}{\partial \Sigma^{-1}}
= \frac{N}{2}\Sigma - \frac{1}{2}\sum_i (x_i - \mu)(x_i - \mu)^T = 0
$$

得到：

$$
\boxed{
\hat{\Sigma} = \frac{1}{N}\sum_i (x_i - \hat{\mu})(x_i - \hat{\mu})^T
}
$$

因此，

$$
\boxed{
\begin{aligned}
\hat{\mu} &= \frac{1}{N}\sum_i x_i,\\
\hat{\Sigma} &= \frac{1}{N}\sum_i (x_i - \hat{\mu})(x_i - \hat{\mu})^T
\end{aligned}
}
$$

