# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# Utils
# --------------------------
def set_seed(seed=42):
    np.random.seed(seed)

def ensure_path(csv_path: str) -> Path:
    p = Path(csv_path)
    if not p.is_file():
        p = Path(__file__).parent / p.name
    if not p.is_file():
        raise FileNotFoundError(f"找不到資料檔：{p.resolve()}")
    print(f"[INFO] 讀取資料：{p.resolve()}")
    return p

def train_val_split(X, y, val_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)
    N = len(X); idx = rng.permutation(N)
    n_val = int(N * val_ratio)
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]

# --------------------------
# Fourier positional features
# --------------------------
class FourierFeatures:
    """
    將 (lon, lat) 以隨機頻率做 sin/cos 映射，增強座標回歸的表達力
    x -> [x, sin(Bx), cos(Bx)]
    """
    def __init__(self, in_dim=2, num_freqs=16, sigma=2.0, seed=42, include_input=True):
        rng = np.random.default_rng(seed)
        # B ~ N(0, sigma^2)
        self.B = rng.normal(loc=0.0, scale=sigma, size=(in_dim, num_freqs))
        self.include_input = include_input

    def transform(self, X):
        # X: (N, 2)
        Z = X @ self.B  # (N, num_freqs)
        feats = [np.sin(Z), np.cos(Z)]  # 兩倍頻寬
        if self.include_input:
            feats = [X] + feats
        return np.concatenate(feats, axis=1)

# --------------------------
# MLP 回歸 + Huber loss + Adam
# --------------------------
class MLPReg:
    def __init__(self, in_dim, h=[256, 256, 128], out_dim=1, seed=42):
        rng = np.random.default_rng(seed)

        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0/(fan_in+fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        dims = [in_dim] + h + [out_dim]
        self.W = []
        self.b = []
        for i in range(len(dims)-1):
            self.W.append(xavier(dims[i], dims[i+1]))
            self.b.append(np.zeros((1, dims[i+1])))

        # Adam 狀態
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0

    @staticmethod
    def tanh(x): return np.tanh(x)

    def forward(self, x, cache=False):
        a = x
        acts = [a]
        zs = []
        L = len(self.W)
        for i in range(L-1):
            z = a @ self.W[i] + self.b[i]
            a = self.tanh(z)
            zs.append(z); acts.append(a)
        # 最後一層線性輸出
        zL = a @ self.W[-1] + self.b[-1]
        y = zL
        if cache:
            zs.append(zL); acts.append(y)
            return y, (acts, zs)
        return y

    @staticmethod
    def huber_loss(y_pred, y_true, delta=1.0):
        # Smooth L1
        r = y_pred - y_true
        abs_r = np.abs(r)
        quad = np.minimum(abs_r, delta)
        lin  = abs_r - quad
        loss = np.mean(0.5 * quad**2 + delta * lin)
        # 對 y_pred 的導數
        grad = np.where(abs_r <= delta, r, delta * np.sign(r)) / y_pred.shape[0]
        return loss, grad

    def backward(self, cache, dL_dy):
        acts, zs = cache
        L = len(self.W)
        grads_W = [None]*L
        grads_b = [None]*L

        # 最後一層（線性）：dz = dL/dy
        delta = dL_dy
        grads_W[-1] = acts[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # 隱藏層
        for i in range(L-2, -1, -1):
            da = delta @ self.W[i+1].T
            dz = da * (1 - np.tanh(zs[i])**2)  # tanh'
            grads_W[i] = acts[i].T @ dz
            grads_b[i] = np.sum(dz, axis=0, keepdims=True)
            delta = dz

        return grads_W, grads_b

    def adam_step(self, grads_W, grads_b, lr=3e-4, wd=1e-6, beta1=0.9, beta2=0.999, eps=1e-8, clip=1.0):
        # 將 W, b 拉成單一陣列來做 Adam 更新（方便）
        params = self.W + self.b
        grads  = grads_W + grads_b

        # 梯度裁剪
        if clip is not None:
            grads = [np.clip(g, -clip, clip) for g in grads]

        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g + wd * p
            self.m[i] = beta1*self.m[i] + (1-beta1)*g
            self.v[i] = beta2*self.v[i] + (1-beta2)*(g*g)
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # 寫回
        nW = len(self.W)
        self.W = params[:nW]
        self.b = params[nW:]

# --------------------------
# Training
# --------------------------
def train_regression(
    csv_path="temperature_regression.csv",
    num_freqs=32, sigma=3.0, include_input=True,  # Fourier 參數
    h=[256, 256, 128],
    epochs=3000, base_lr=1e-3, min_lr=2e-4, batch=128,
    patience=400, seed=42, val_ratio=0.25, weight_decay=1e-6,
    huber_delta=1.0
):
    p = ensure_path(csv_path)
    df = pd.read_csv(p)

    if "value" not in df.columns:
        raise ValueError("CSV 需包含欄位：longitude, latitude, value")
    df = df[df["value"] > -999]

    X_raw = df[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    y = df[["value"]].to_numpy(dtype=np.float64)

    # 標準化輸入與輸出
    x_mu, x_sd = X_raw.mean(0, keepdims=True), X_raw.std(0, keepdims=True) + 1e-8
    y_mu, y_sd = y.mean(0, keepdims=True), y.std(0, keepdims=True) + 1e-8
    Xn = (X_raw - x_mu) / x_sd
    yn = (y - y_mu) / y_sd

    # Fourier features
    fe = FourierFeatures(in_dim=2, num_freqs=num_freqs, sigma=sigma, seed=seed, include_input=include_input)
    Z = fe.transform(Xn)  # (N, D_feat)

    Xtr, ytr, Xva, yva = train_val_split(Z, yn, val_ratio=val_ratio, seed=seed)

    net = MLPReg(in_dim=Z.shape[1], h=h, out_dim=1, seed=seed)
    rng = np.random.default_rng(seed)

    tr_hist, va_hist = [], []
    best_va = np.inf; best_state = None; wait = 0

    # Cosine annealing LR
    def lr_at(ep):
        if epochs <= 1: return base_lr
        cos = 0.5 * (1 + np.cos(np.pi * ep / epochs))
        return min_lr + (base_lr - min_lr) * cos

    for ep in range(1, epochs + 1):
        lr = lr_at(ep)

        # mini-batch
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            j = idx[i:i+batch]
            xb, yb = Xtr[j], ytr[j]
            y_pred, cache = net.forward(xb, cache=True)
            loss, dL_dy = net.huber_loss(y_pred, yb, delta=huber_delta)
            grads_W, grads_b = net.backward(cache, dL_dy)
            net.adam_step(grads_W, grads_b, lr=lr, wd=weight_decay, clip=1.0)

        # log
        ypt = net.forward(Xtr)
        loss_tr, _ = net.huber_loss(ypt, ytr, delta=huber_delta)
        ypv = net.forward(Xva)
        loss_va, _ = net.huber_loss(ypv, yva, delta=huber_delta)
        tr_hist.append(loss_tr); va_hist.append(loss_va)

        if loss_va < best_va - 1e-6:
            best_va = loss_va
            best_state = ( [w.copy() for w in net.W], [b.copy() for b in net.b] )
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[INFO] Early stopped at epoch {ep}")
                break

    if best_state is not None:
        W, B = best_state
        net.W = W; net.b = B

    norm = {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd, "fe": fe}
    return net, (np.array(tr_hist), np.array(va_hist)), norm, (X_raw, y)

# --------------------------
# Evaluation
# --------------------------
def evaluate_and_plot_reg(net, norm, X_raw, y, save_prefix="reg_ff"):
    # 建構驗證集（重新切一次只是為了畫圖一致；不影響指標正確性）
    rng = np.random.default_rng(0)
    N = len(X_raw); idx = rng.permutation(N); n_val = int(N*0.25)
    Xva_raw, yva = X_raw[idx[:n_val]], y[idx[:n_val]]

    # transform
    Xva_n = (Xva_raw - norm["x_mu"]) / norm["x_sd"]
    Zva   = norm["fe"].transform(Xva_n)

    # predict (回到原始 °C)
    yva_pred_n = net.forward(Zva)
    yva_pred   = yva_pred_n * norm["y_sd"] + norm["y_mu"]

    mse = mean_squared_error(yva, yva_pred)
    mae = mean_absolute_error(yva, yva_pred)
    r2  = r2_score(yva, yva_pred)
    print(f"Validation MSE : {mse:.4f}")
    print(f"Validation MAE : {mae:.4f}")
    print(f"Validation R^2 : {r2:.4f}")

    # y vs y_hat
    plt.figure(figsize=(5.2,4.6))
    plt.scatter(yva, yva_pred, s=10, alpha=0.6)
    lo, hi = np.min([yva, yva_pred]), np.max([yva, yva_pred])
    plt.plot([lo,hi],[lo,hi],"--",lw=1)
    plt.xlabel("True Temp (°C)"); plt.ylabel("Predicted (°C)")
    plt.title("Validation: y vs y_hat (Fourier-MLP)")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_scatter.png", dpi=150); plt.show()

    # 殘差
    resid = (yva_pred - yva).ravel()
    plt.figure(figsize=(5.2,4.6))
    sns.histplot(resid, bins=40, kde=True)
    plt.title("Residuals (pred - true)")
    plt.xlabel("Residual (°C)"); plt.tight_layout()
    plt.savefig(f"{save_prefix}_resid.png", dpi=150); plt.show()

    return mse, mae, r2

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    set_seed(111652006)

    net, (tr_hist, va_hist), norm, (X_raw, y) = train_regression(
        csv_path="temperature_regression.csv",
        num_freqs=24, sigma=3.0, include_input=True,
        h=[128, 128, 64],
        epochs=2000, base_lr=1e-3, min_lr=1e-4, batch=256,
        patience=250, weight_decay=3e-6, huber_delta=2.0
    )

    # Loss curve
    plt.figure()
    plt.plot(tr_hist, label="train Huber")
    plt.plot(va_hist, label="val Huber")
    plt.xlabel("epoch"); plt.ylabel("Huber (on normalized y)")
    plt.title("Training / Validation Loss (Regression with Fourier features)")
    plt.legend(); plt.tight_layout(); plt.savefig("reg_ff_loss.png", dpi=150); plt.show()

    # Eval
    evaluate_and_plot_reg(net, norm, X_raw, y, save_prefix="reg_ff")
