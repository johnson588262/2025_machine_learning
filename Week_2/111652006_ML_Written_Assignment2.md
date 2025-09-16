# Assignment 2
## 問題一 透過演算法計算\nabla a^{[L]}(x) when n_L = 1
反正就是要走一遍反向傳播
## 目標
計算 the gradient of the scalar output $a^{[L]}$ w.r.t. the input $x=a^{[1]}$ :

$$
\nabla_x a^{[L]}(x) \in \mathbb{R}^{n_1}.
$$

## Network 
For $l=2,\dots,L$ :

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]},\qquad a^{[l]} = f^{[l]}(z^{[l]}),
$$

with $a^{[1]}=x$ and $n_L=1$ (scalar output).

## Algorithm 

**Input:** $x=a^{[1]}$ , parameters $\{W^{[l]},b^{[l]}\}_{l=2}^L$ , activations $f^{[l]}$ .  
**Output:** $g=\nabla_x a^{[L]}(x)\in\mathbb{R}^{n_1}$.

1. **Forward pass** (store intermediates):  
   For $l=2,\dots,L$ :

   $$
   z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]},\quad a^{[l]} = f^{[l]}(z^{[l]}).
   $$

2. **Initialize output sensitivity (scalar):**

   $$
   \delta^{[L]} = \frac{\partial a^{[L]}}{\partial z^{[L]}}
                 = f^{[L]\prime}(z^{[L]}).
   $$

3. **Backward pass to layer 2:**
   For $l=L-1,L-2,\dots,2$ :

   $$
   \delta^{[l]} = (W^{[l+1]})^\top \delta^{[l+1]} \odot f^{[l]\prime}(z^{[l]}).
   $$

   (⊙ = 外積; $f^{[l]\prime}$ 每個元素各自求導)

4. **Gradient w.r.t. input** $x=a^{[1]}$ :

   $$
   g = \nabla_x a^{[L]}(x) = (W^{[2]})^\top \delta^{[2]}.
   $$

## 問題二
 MSE 可以解釋成高斯假設下的最大似然估計。如果誤差分布不是 Gaussian（例如 Laplace 分布 student-t分布），那 loss function 會變成什麼？ 還會是一樣的嗎?

