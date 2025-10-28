
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

# --------------------------
# Utility
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

def train_val_split(X, y, val_ratio=0.25, seed=42, stratify=True):
    rng = np.random.default_rng(seed)
    N = len(X)
    idx = np.arange(N)
    if stratify:
        idx0 = idx[y.flatten() == 0]
        idx1 = idx[y.flatten() == 1]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        n0_val = int(len(idx0) * val_ratio)
        n1_val = int(len(idx1) * val_ratio)
        val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
        tr_idx  = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    else:
        rng.shuffle(idx)
        n_val = int(N * val_ratio)
        val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

# --------------------------
# MLP with Adam + weighted BCE
# --------------------------
class MLP:
    def __init__(self, in_dim=2, h1=64, h2=64, out_dim=1, seed=42, p_pos=0.5):
        rng = np.random.default_rng(seed)
        def xavier(shape):
            fan_in, fan_out = shape
            limit = np.sqrt(6.0/(fan_in+fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))
        self.W1 = xavier((in_dim, h1)); self.b1 = np.zeros((1,h1))
        self.W2 = xavier((h1, h2));     self.b2 = np.zeros((1,h2))
        self.W3 = xavier((h2, out_dim))
        # bias 初始化為 logit(prior)
        self.b3 = np.array([[np.log(p_pos/(1-p_pos))]])
        # Adam 狀態
        self._adam = {k:{"m":np.zeros_like(getattr(self,k)),
                         "v":np.zeros_like(getattr(self,k))} 
                      for k in ["W1","b1","W2","b2","W3","b3"]}
        self._t = 0

    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def sigmoid(x): return 1/(1+np.exp(-x))

    def forward(self, x, cache=False):
        z1 = x@self.W1 + self.b1; a1 = self.tanh(z1)
        z2 = a1@self.W2 + self.b2; a2 = self.tanh(z2)
        z3 = a2@self.W3 + self.b3; y = self.sigmoid(z3)
        if cache: return y,(x,z1,a1,z2,a2,z3)
        return y

    def compute_loss(self, y_true, y_pred, w0, w1, eps=1e-9):
        return -np.mean(w1*y_true*np.log(y_pred+eps) + w0*(1-y_true)*np.log(1-y_pred+eps))

    def backward(self, cache, y_pred, y_true, w0, w1):
        x,z1,a1,z2,a2,z3 = cache
        N = x.shape[0]
        w = w1 * y_true + w0 * (1.0 - y_true)
        dz3 = w * (y_pred - y_true) / N
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3,axis=0,keepdims=True)
        da2 = dz3@self.W3.T
        dz2 = da2*(1-np.tanh(z2)**2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2,axis=0,keepdims=True)
        da1 = dz2@self.W2.T
        dz1 = da1*(1-np.tanh(z1)**2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1,axis=0,keepdims=True)
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}

    def adam_step(self, grads, lr=1e-3, wd=1e-6, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k,g in grads.items():
            p = getattr(self,k)
            g = g + wd*p
            self._adam[k]["m"] = beta1*self._adam[k]["m"]+(1-beta1)*g
            self._adam[k]["v"] = beta2*self._adam[k]["v"]+(1-beta2)*(g*g)
            m_hat = self._adam[k]["m"]/(1-beta1**self._t)
            v_hat = self._adam[k]["v"]/(1-beta2**self._t)
            p -= lr*m_hat/(np.sqrt(v_hat)+eps)
            setattr(self,k,p)

# --------------------------
# Training loop
# --------------------------
def train(csv_path="temperature_classification.csv", h1=64,h2=64,epochs=2000,lr=1e-3,batch=32,patience=300,seed=42,val_ratio=0.25):
    p = ensure_path(csv_path)
    df = pd.read_csv(p)
    X = df[["longitude","latitude"]].to_numpy(dtype=np.float64)
    y = df[["label"]].to_numpy(dtype=np.float64)
    Xtr,ytr,Xva,yva = train_val_split(X,y,val_ratio=val_ratio,seed=seed,stratify=True)
    # 標準化
    mu,sd = Xtr.mean(0,keepdims=True),Xtr.std(0,keepdims=True)+1e-8
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    # class weight
    N=len(ytr); N0=np.sum(ytr==0); N1=np.sum(ytr==1)
    w0=N/(2*N0); w1=N/(2*N1)
    p_pos = N1/N
    net = MLP(2,h1,h2,1,seed=seed,p_pos=p_pos)

    tr_hist,va_hist=[],[]
    best_va=np.inf; best_state=None; wait=0
    rng=np.random.default_rng(seed)

    for ep in range(1,epochs+1):
        # mini-batch
        idx = rng.permutation(len(Xtr))
        for i in range(0,len(Xtr),batch):
            j = idx[i:i+batch]; xb,yb=Xtr[j],ytr[j]
            yp,cache=net.forward(xb,cache=True)
            loss = net.compute_loss(yb,yp,w0,w1)
            grads=net.backward(cache,yp,yb,w0,w1)
            net.adam_step(grads,lr=lr)
        # log
        ypt=net.forward(Xtr); ypv=net.forward(Xva)
        tr_loss=net.compute_loss(ytr,ypt,w0,w1)
        va_loss=net.compute_loss(yva,ypv,w0,w1)
        tr_hist.append(tr_loss); va_hist.append(va_loss)
        if va_loss<best_va-1e-6:
            best_va=va_loss
            best_state={k:getattr(net,k).copy() for k in ["W1","b1","W2","b2","W3","b3"]}
            wait=0
        else: wait+=1
        if wait>=patience: 
            print(f"[INFO] Early stopped at epoch {ep}")
            break
    if best_state: 
        for k,v in best_state.items(): setattr(net,k,v)
    return net,(tr_hist,va_hist),(Xtr,ytr,Xva,yva)

# --------------------------
# Evaluation
# --------------------------
def evaluate_and_plot(net, Xtr, ytr, Xva, yva, save_prefix=None):
    y_pred = net.forward(Xva)

    # Accuracy 和 Confusion Matrix 用二元標籤
    acc = accuracy_score(yva, (y_pred > 0.5).astype(int))
    auc = roc_auc_score(yva, y_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(yva, (y_pred > 0.5).astype(int))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(yva, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.show()

    return acc, auc

# --------------------------
# Main
# --------------------------
if __name__=="__main__":
    set_seed(111652006)
    net,(tr_hist,va_hist),(Xtr,ytr,Xva,yva)=train("temperature_classification.csv")
    plt.figure()
    plt.plot(tr_hist, label="train loss")
    plt.plot(va_hist, label="val loss")
    plt.xlabel("epoch"); plt.ylabel("weighted BCE"); plt.legend()
    plt.title("Training / Validation Loss")
    plt.savefig("cls_loss_curve.png", dpi=150)
    plt.show()

    evaluate_and_plot(net, Xtr, ytr, Xva, yva, save_prefix="cls")