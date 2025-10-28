# runge_value_and_derivative.py
import numpy as np
import matplotlib.pyplot as plt

# ========= Config =========
SEED = 111652006                 # 固定隨機種子
H1, H2 = 128, 128                # 隱藏層寬度
LR = 1e-2                        # 學習率（SGD）
WD = 1e-6                        # L2 weight decay
BATCH = 16                       # mini-batch size
EPOCHS = 4000                    # 訓練輪數上限
PATIENCE = 400                   # 早停等待
ALPHA = 1.0                      # derivative loss 權重（L = L_f + ALPHA * L_d）
H_DIFF = 1e-3                    # 導數近似的步長 h

rng = np.random.default_rng(SEED)

# ========= Data =========
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_prime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

def make_split(n_train=200, n_val=200):
    x_tr = rng.uniform(-1.0, 1.0, size=(n_train, 1))
    y_tr = runge(x_tr)
    x_va = rng.uniform(-1.0, 1.0, size=(n_val, 1))
    y_va = runge(x_va)
    return x_tr, y_tr, x_va, y_va

# dense grid for plotting/eval
x_grid = np.linspace(-1.0, 1.0, 1000)[:, None]
y_true_grid  = runge(x_grid)
dy_true_grid = runge_prime(x_grid)

# ========= Model =========
class MLP:
    def __init__(self, in_dim=1, h1=64, h2=64, out_dim=1):
        def xavier(shape):
            fan_in, fan_out = shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))
        self.W1 = xavier((in_dim, h1));   self.b1 = np.zeros((1, h1))
        self.W2 = xavier((h1, h2));       self.b2 = np.zeros((1, h2))
        self.W3 = xavier((h2, out_dim));  self.b3 = np.zeros((1, out_dim))

    @staticmethod
    def tanh(x): return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        y = np.tanh(x)
        return 1.0 - y*y

    def forward(self, x, cache=False):
        z1 = x @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        y  = z3  # linear output
        if cache:
            return y, (x, z1, a1, z2, a2, z3)
        return y

    # 允許自定義上游梯度 dL/dy 的反傳（用於合併 derivative loss）
    def backward_from_upstream(self, cache, upstream):
        x, z1, a1, z2, a2, z3 = cache

        # output layer (linear)
        dz3 = upstream                          # (N,1)
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # hidden 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.tanh_prime(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # hidden 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.tanh_prime(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1,
                "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3}

    def sgd_step(self, grads, lr=1e-2, weight_decay=0.0):
        # 簡單 SGD + L2
        for k, g in grads.items():
            p = getattr(self, k)
            p -= lr * (g / 1.0 + weight_decay * p)  # /1.0 保持一致性
            setattr(self, k, p)

# ========= Training utils =========
def minibatches(X, Y, batch=32, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle: rng.shuffle(idx)
    for i in range(0, N, batch):
        j = idx[i:i+batch]
        yield X[j], Y[j]

def add_grads(*grad_dicts):
    out = {k: 0 for k in grad_dicts[0].keys()}
    for gd in grad_dicts:
        for k, v in gd.items():
            out[k] = out[k] + v
    return out

def train_model(h1=H1, h2=H2, epochs=EPOCHS, lr=LR, batch=BATCH,
                patience=PATIENCE, wd=WD, alpha=ALPHA, h=H_DIFF):
    x_tr, y_tr, x_va, y_va = make_split()
    net = MLP(1, h1, h2, 1)

    tr_hist_f, tr_hist_d = [], []
    va_hist_f, va_hist_d = [], []

    best_val = np.inf
    best_state = None
    wait = 0

    for ep in range(1, epochs+1):
        # -------- Train --------
        for xb, yb in minibatches(x_tr, y_tr, batch=batch, shuffle=True):
            N = xb.shape[0]

            # (1) function part
            y, cache = net.forward(xb, cache=True)
            upstream_f = (2.0 / N) * (y - yb)                 # dL_f/dy(x)
            grads_f = net.backward_from_upstream(cache, upstream_f)

            # (2) derivative part (central difference)
            dy_true = runge_prime(xb)
            x_plus,  x_minus  = xb + h, xb - h
            y_p, cache_p = net.forward(x_plus,  cache=True)
            y_m, cache_m = net.forward(x_minus, cache=True)
            dy_hat = (y_p - y_m) / (2.0*h)
            resid  = dy_hat - dy_true
            coeff  = (2.0 / N) * (1.0 / (2.0*h))
            upstream_p = coeff * resid                        # dL_d/dy(x+h)
            upstream_m = -coeff * resid                       # dL_d/dy(x-h)
            grads_p = net.backward_from_upstream(cache_p, upstream_p)
            grads_m = net.backward_from_upstream(cache_m, upstream_m)

            # 合併梯度並更新
            grads = add_grads(grads_f, {k: alpha * v for k, v in add_grads(grads_p, grads_m).items()})
            net.sgd_step(grads, lr=lr, weight_decay=wd)

            # 記錄 batch loss（用於可視化）
            loss_f = np.mean((y - yb)**2)
            loss_d = np.mean((dy_hat - dy_true)**2)
            tr_hist_f.append(loss_f); tr_hist_d.append(loss_d)

        # -------- Validation --------
        with np.errstate(all='ignore'):
            yv, _ = net.forward(x_va, cache=True)

            # derivative on val
            xvp, xvm = x_va + h, x_va - h
            yvp = net.forward(xvp)
            yvm = net.forward(xvm)
            dyv_hat = (yvp - yvm) / (2.0*h)

            dyv_true = runge_prime(x_va)

            val_f = np.mean((yv - y_va)**2)
            val_d = np.mean((dyv_hat - dyv_true)**2)

        va_hist_f.append(val_f); va_hist_d.append(val_d)

        val_total = val_f + alpha * val_d
        if val_total < best_val - 1e-9:
            best_val = val_total
            best_state = {k: getattr(net, k).copy() for k in net.__dict__ if k.startswith(("W","b"))}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        for k, v in best_state.items():
            setattr(net, k, v)

    return net, (np.array(tr_hist_f), np.array(tr_hist_d)), (np.array(va_hist_f), np.array(va_hist_d)), (x_tr, y_tr, x_va, y_va)

# ========= Run =========
if __name__ == "__main__":
    net, (trF, trD), (vaF, vaD), (x_tr, y_tr, x_va, y_va) = train_model()

    # evaluate on dense grid
    y_pred = net.forward(x_grid)
    # derivative via central difference on dense grid
    yg_p = net.forward(x_grid + H_DIFF)
    yg_m = net.forward(x_grid - H_DIFF)
    dy_pred = (yg_p - yg_m) / (2.0*H_DIFF)

    mse_f   = np.mean((y_pred - y_true_grid)**2)
    max_f   = np.max(np.abs(y_pred - y_true_grid))
    mse_df  = np.mean((dy_pred - dy_true_grid)**2)
    max_df  = np.max(np.abs(dy_pred - dy_true_grid))

    print(f"MSE(f)      : {mse_f:.6e}   MaxErr(f)  : {max_f:.6e}")
    print(f"MSE(f')     : {mse_df:.6e}  MaxErr(f') : {max_df:.6e}")

    # ----- Plot: function -----
    plt.figure()
    plt.plot(x_grid[:,0], y_true_grid[:,0], label="True f(x)")
    plt.plot(x_grid[:,0], y_pred[:,0], label="NN f(x)")
    plt.scatter(x_tr[:,0], y_tr[:,0], s=10, alpha=0.35, label="Train samples")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Runge function approximation (value + derivative loss)")
    plt.legend(); plt.tight_layout(); plt.show()

    # ----- Plot: derivative -----
    plt.figure()
    plt.plot(x_grid[:,0], dy_true_grid[:,0], label="True f'(x)")
    plt.plot(x_grid[:,0], dy_pred[:,0], label="NN f'(x)")
    plt.xlabel("x"); plt.ylabel("dy/dx")
    plt.title("Derivative approximation (central difference)")
    plt.legend(); plt.tight_layout(); plt.show()

    # ----- Plot: losses -----
    plt.figure()
    plt.plot(trF, label="train MSE(f)")
    plt.plot(trD, label="train MSE(f')")
    plt.plot(vaF, label="val MSE(f)")
    plt.plot(vaD, label="val MSE(f')")
    plt.xlabel("iteration / epoch-batches"); plt.ylabel("MSE")
    plt.title("Training/Validation losses")
    plt.legend(); plt.tight_layout(); plt.show()
