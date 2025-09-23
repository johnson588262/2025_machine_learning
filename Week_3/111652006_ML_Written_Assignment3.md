# Assignment 3
## 問題一 解釋Lemma

**Paper:** De Ryck, Lanthaler, and Mishra (2021), *On the Approximation of Functions by Tanh Neural Networks*

---

# Lemma 3.1 與 Lemma 3.2 的說明報告

**來源：** De Ryck, Lanthaler, and Mishra (2021),  
*On the Approximation of Functions by tanh Neural Networks*

---

在 approximation theory 裡，多項式很重要，因為根據 **Weierstrass approximation theorem**，任何連續函數都能被多項式任意精度逼近。  
所以如果能證明 **tanh 神經網路能近似所有 monomials（$x^n$）**，那麼就能近似任何連續函數。

簡單來說 **Lemma 3.1** 和 **Lemma 3.2** 的重點就是：

- **Lemma 3.1**：說明 tanh 網路可以近似 **奇數次單項式**（$x, x^3, x^5, \dots$）。
- **Lemma 3.2**：進一步補上 **偶數次單項式**（$x^2, x^4, \dots$）。

合起來 → 任何多項式都能被 tanh Network逼近。

---

## Lemma 3.1（奇數次 monomials）

### 命題
對任意正整數 $s$（奇數），在區間 $[-M,M]$ 上，存在一個 **一層隱藏層（shallow）** 的 tanh 神經網路 $\Psi_{s,\varepsilon}$，能同時逼近所有 $p\leq s$ 的奇數次 monomials（$x, x^3, \dots, x^s$），在 **Sobolev norm $W^{k,\infty}$** 下誤差 $\leq \varepsilon$。

### 證明的一些重點
1. **中心差分運算子 (central finite difference)**  
   對函數 $f$ 定義
   $$
   \delta_h^p[f](x) = \sum_{i=0}^p (-1)^i \binom{p}{i} f\!\Big(x + \Big(\tfrac{p}{2}-i\Big)h\Big).
   $$
   因為那個 $(-1)^i$ 的存在，所以把 $f$ 泰勒展開時會消掉 $f$ 的低階項( $\sum_{i=0}^p (-1)^i \binom{p}{i} \left(\tfrac{p}{2}-i\right)^m$ )，留下與 $f^{(p)}(0)$ 成正比的項。

2. **套用 $f=\tanh$**  
   定義
   $$
   \hat f_{q,h}(y) = \frac{\delta_{hy}^q }{\tanh^{(q)}(0)\,h^q}.
   $$
   因為 $\tanh$ 是奇函數，對奇數 $q$ 有 $\tanh^{(q)}(0)\neq 0$。  
   這樣設計可以讓主項對齊 $y^q$。

3. **error控制**  
   用 Taylor 展開，誤差項是 $O(h^2 y^{q+2})$。  
   選 $h \sim \sqrt{\varepsilon}$，就能把誤差壓到 $\varepsilon$。


---

## Lemma 3.2（偶數次 monomials）

### 命題
同樣在區間 $[-M,M]$ 上，對於任何偶數次 monomials（$x^2, x^4, \dots$），也能構造淺層 tanh 網路逼近到任意精度。

### 文章提到的證明想法
1. **困難點**  
   若直接模仿 Lemma 3.1，需要找到 $\tanh^{(q)}(x_*)$ 在某點的下界，但對偶數 $q$ 不容易。

2. **繞道：三點平移消奇留偶**  
   考慮
   $$
   A_\alpha(x) = g(x+\alpha) - 2g(x) + g(x-\alpha).
   $$
   如果 $g$ 是奇函數，則這樣的組合會抵消掉所有奇數項，只留下偶數項。  
   這就是 Lemma 3.2 的核心技巧。

3. **結論**  
   把 Lemma 3.1 的「奇次近似」$g(x)$ 平移後組合，就能得到偶次 monomials 的近似。  
   如此一來，所有 $p \leq s$ 的 monomials（奇、偶）都能同時在 $W^{k,\infty}$ 意義下近似。

