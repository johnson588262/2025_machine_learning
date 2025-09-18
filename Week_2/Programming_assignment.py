import numpy as np
import matplotlib.pyplot as plt

#要製造隨機性 同時保證結果可重現 這邊seed放一下自己學號
rng = np.random.default_rng(111652006)

# ----- data -----
#題目的runge func
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

#隨機產生訓練集和驗證籍資料
def make_split(n_train=200, n_val=200):
    x_tr = rng.uniform(-1.0, 1.0, size=(n_train, 1))
    y_tr = runge(x_tr)
    x_va = rng.uniform(-1.0, 1.0, size=(n_val, 1))
    y_va = runge(x_va)
    return x_tr, y_tr, x_va, y_va

# 劃1000個等距點來劃真實的函數曲線和神經網路預測 計算誤差
x_grid = np.linspace(-1.0, 1.0, 1000)[:, None]
y_true = runge(x_grid)

# ----- model -----
class MLP:
    def __init__(self, in_dim=1, h1=128, h2=128, out_dim=1):
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
    def tanh_prime(x): return 1.0 - np.tanh(x)**2

    #幫NN的每個重要部分令好參數
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

    def loss_mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def backward(self, cache, y_pred, y_true):
        x, z1, a1, z2, a2, z3 = cache
        N = x.shape[0]

        dy = (2.0 / N) * (y_pred - y_true)  # dL/dy

        dz3 = dy
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.tanh_prime(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.tanh_prime(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1,
                 "W2": dW2, "b2": db2,
                 "W3": dW3, "b3": db3}
        return grads

    def sgd_step(self, grads, lr=1e-2, weight_decay=1e-4):
        for k, g in grads.items():
            p = getattr(self, k)
            g = g + weight_decay * p
            setattr(self, k, p - lr * g)

# ----- training -----
def minibatches(X, Y, batch=32, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle: rng.shuffle(idx)
    for i in range(0, N, batch):
        j = idx[i:i+batch]
        yield X[j], Y[j]

def train_model(h1=128, h2=128, epochs=4000, lr=1e-2, batch=16, patience=400, wd=1e-6):
    x_tr, y_tr, x_va, y_va = make_split()
    net = MLP(1, h1, h2, 1)

    tr_hist, va_hist = [], []
    best_va = np.inf
    best_state = None
    wait = 0

    for ep in range(1, epochs+1):
        # train
        for xb, yb in minibatches(x_tr, y_tr, batch=batch, shuffle=True):
            yp, cache = net.forward(xb, cache=True)
            grads = net.backward(cache, yp, yb)
            net.sgd_step(grads, lr=lr, weight_decay=wd)

        # log
        ypt = net.forward(x_tr)
        ypv = net.forward(x_va)
        tr_loss = net.loss_mse(ypt, y_tr)
        va_loss = net.loss_mse(ypv, y_va)
        tr_hist.append(tr_loss); va_hist.append(va_loss)

        # early stopping
        if va_loss < best_va - 1e-9:
            best_va = va_loss
            best_state = {k: getattr(net, k).copy() for k in net.__dict__ if k.startswith(("W","b"))}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        for k, v in best_state.items():
            setattr(net, k, v)

    return net, (np.array(tr_hist), np.array(va_hist)), (x_tr, y_tr, x_va, y_va)

# ----- run -----
if __name__ == "__main__":
    net, (tr_hist, va_hist), (x_tr, y_tr, x_va, y_va) = train_model()

    y_pred = net.forward(x_grid)
    mse = np.mean((y_pred - y_true)**2)
    max_err = np.max(np.abs(y_pred - y_true))
    print(f"MSE on dense grid: {mse:.6e}")
    print(f"Max error (L_inf): {max_err:.6e}")

    # plot: function vs prediction
    plt.figure()
    plt.plot(x_grid[:,0], y_true[:,0], label="True f(x)")
    plt.plot(x_grid[:,0], y_pred[:,0], label="NN prediction")
    plt.scatter(x_tr[:,0], y_tr[:,0], s=10, alpha=0.4, label="Train samples")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Runge function approximation (SGD)")
    plt.legend(); plt.tight_layout(); plt.show()

    # plot: loss curves
    plt.figure()
    plt.plot(tr_hist, label="train MSE")
    plt.plot(va_hist, label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Training / validation loss (SGD)")
    plt.legend(); plt.tight_layout(); plt.show()
