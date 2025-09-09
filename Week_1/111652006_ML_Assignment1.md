# Assignment 1
## 問題一 利用Gradient Descent Method計算 theta^1
已經有原始的 $\theta^0 = (b,w_1,w_2) = (4,5,6)$，且根據題意

$$
\text{Model: } h(x_1, x_2) = \sigma(b + w_1x_1 + w_2x_2), \quad \sigma(z) = \frac{1}{1+e^{-z}}.
$$ 

由Gradient descent algorithm得知

$$
\theta^{n+1} = \theta^n - \alpha\nabla_{\theta}\text{Loss}.
$$

對於MSE Loss，可以寫成

$$ 
\theta^{n+1} = \theta^n + 2\alpha\,\left[\frac{1}{m}\sum^m_{i=1} \left(y^i - h(x^i_1, x^i_2)\right)\nabla_\theta h\right].
$$

因為要求 $\theta^1$，丟進去式子就會得到

$$
\theta^{1} = \theta^{0} + 2\alpha \,(y - h(x_1,x_2)) \, \nabla_\theta h.
$$

其中

$$
\theta^{0} = (b^0, w_1^0, w_2^0) = (4, 5, 6), \quad (x_1, x_2, y) = (1, 2, 3).
$$

接著我想要計算 $\sigma(b + w_1x_1 + w_2x_2)$ 這部分，所以

$$
\text{Let } z = b^0 + w_1^0 x_1 + w_2^0 x_2\text{, then } z = 4 + 5\cdot 1 + 6\cdot 2 = 21.
$$

$$
h(x_1,x_2) = \sigma(z) = \sigma(21) = \frac{1}{1+e^{-21}} \\
= h \text{(反正沒要求要算出來偷懶一下)}
$$

因此

$$
2\alpha \,(y - h(x_1,x_2)) = 2\alpha \,(3 - h).
$$

接下來是要處理 $\nabla_\theta h$ ，所以要拿 $h(x_1,x_2)$ 去對 $(b,w_1,w_2)$ 分別求偏導，但在此之前，可以先運算一下 $\sigma(z)$ 對 $z$ 求導

$$
\sigma'(z) = \frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right) 
= \frac{e^{-z}}{(1+e^{-z})^2}.
$$

因為

$$
\sigma(z) = \frac{1}{1+e^{-z}}, 
\quad 1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}},
$$

所以

$$
\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr).
$$

透過Chian Rule，我們就可以很快地寫出

$$
\frac{\partial h}{\partial \theta_j} 
= \sigma'(z)\cdot \frac{\partial z}{\partial \theta_j}.
$$

- **對 \(b\)：**

$$
\frac{\partial z}{\partial b} = 1 
\quad\Rightarrow\quad 
\frac{\partial h}{\partial b} = h(1-h).
$$

- **對 \(w_1\)：**

$$
\frac{\partial z}{\partial w_1} = x_1 
\quad\Rightarrow\quad 
\frac{\partial h}{\partial w_1} = h(1-h)\,x_1.
$$

- **對 \(w_2\)：**

$$
\frac{\partial z}{\partial w_2} = x_2 
\quad\Rightarrow\quad 
\frac{\partial h}{\partial w_2} = h(1-h)\,x_2.
$$

因此

$$
\begin{aligned}
\nabla_\theta h &= (h(1-h),\; h(1-h)x_1,\; h(1-h)x_2 )\\
                &= (h(1-h),\; h(1-h),\; 2h(1-h) ).
\end{aligned}
$$

把上面的東西收集起來就會得到

$$
\begin{aligned}
\theta^{1} &= \theta^{0} + 2\alpha \,(y - h(x_1,x_2)) \, \nabla_\theta h \\
            &= (4,5,6) + 2\alpha(3-h)(h(1-h),\; h(1-h),\; 2h(1-h) ) \\
            &= (b^1,w_1^1,w_2^1).
\end{aligned}
$$

其中

$$
\begin{aligned}
b^1   &= 4 + 2\alpha \,(3 - h)\,h(1-h), \\
w_1^1 &= 5 + 2\alpha \,(3 - h)\,h(1-h)\cdot 1, \\
w_2^1 &= 6 + 2\alpha \,(3 - h)\,h(1-h)\cdot 2,
\end{aligned}
$$

## 問題二(a) k階導的Sigmoid function
在上一題就有偷偷的留了這題的起頭

$$
\sigma'(z) = \frac{d}{dz}(\frac{1}{1+e^{-z}}) 
= \frac{e^{-z}}{(1+e^{-z})^2}.
$$

因為

$$
\sigma(z) = \frac{1}{1+e^{-z}}, 
\quad 1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}},
$$

所以

$$
\sigma'(z) = \sigma(z)(1-\sigma(z)).
$$

- **二階導 \sigma''(z)** 

根據Product Rule

$$
\begin{aligned}
\sigma''(z) &= \sigma'(z)(1-\sigma(z))+\sigma(z)(-\sigma'(z)) \\
            &= \sigma'(z)(1=2\sigma(z)) \\
            &= \sigma(z)(1-\sigma(z))(1=2\sigma(z)).
\end{aligned}
$$

- **三階導 \sigma'''(z)**

本質上也是用Product Rule，只不過因為有三項乘積比較麻煩

$$
\begin{aligned}
\sigma'''(z) &= \frac{d}{dx}[\sigma(x)(1-\sigma(x))(1 - 2\sigma(x))] \\
            &= \sigma(x)(1-\sigma(x))[(1-2\sigma(x))^2-2\sigma(x)(1-\sigma(x)) ]\\
            &= \sigma(x)(1-\sigma(x))\bigl(1 - 6\sigma(x) + 6\sigma(x)^2 \bigr).
\end{aligned}
$$

## 問題二(b) Sigmoid function 跟 Hyperbolic function的關係
令Sigmoid function 為 $\sigma(z)$ 

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$ 

且 Hyperbolic function為 $tanh(z)$

$$ 
tanh(z)= \frac{e^z - e^{-z}}{e^z + e^{-z}}.
$$

對 $\sigma(z)$ 分子分母同乘 $e^{\frac{z}{2}}$ :

$$
\begin{aligned}
\sigma(z) &= \frac{e^\frac{z}{2}}{e^\frac{z}{2}+e^\frac{-z}{2}} \\
          &= \frac{1}{2}( \frac{e^{z/2}+e^{-z/2}}{e^{z/2}+e^{-z/2}}+\frac{e^{z/2}-e^{-z/2}}{e^{z/2}+e^{-z/2}}) \\
          &= \frac{1}{2}(1+\tanh(\frac{z}{2})).
\end{aligned}
$$

由 $\sigma(z)= \frac{1}{2}(1+\tanh(\frac{z}{2}))$ 可知，Sigmoid Function 就是 Hyperbolic Function的平移縮放。

## 問題三
我比較好奇要如何決定Mini-Batch Gradient Descent( $m<M$ )要如何挑選合適的batch size，選太小又會像SGD一樣震盪，選太大運算起來又太慢，選擇的依據是根據算力嗎?還是要根據 $\alpha$ 值去調整，有甚麼比較制式化的作法嗎?
