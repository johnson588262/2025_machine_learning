import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_path(csv_path: str) -> Path:
    p = Path(csv_path)
    if not p.is_file():
        p = Path(__file__).parent / p.name
    if not p.is_file():
        raise FileNotFoundError(f"找不到資料檔：{p.resolve()}")
    print(f"[INFO] 讀取資料：{p.resolve()}")
    return p

path = ensure_path("temperature_classification.csv")
df = pd.read_csv(path)
print(df.head())

# 假設資料格式：longitude, latitude, label
X = df[['longitude', 'latitude']].values
y = df['label'].values

# train/test 分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class GDA:
    def __init__(self):
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
        self.phi = None

    def fit(self, X, y):
        # 分類 0/1
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)
        self.phi = len(X1) / len(X)
        # 共用協方差
        sigma = ((X0 - self.mu0).T @ (X0 - self.mu0) + (X1 - self.mu1).T @ (X1 - self.mu1)) / len(X)
        self.sigma = sigma
        self.invSigma = np.linalg.inv(sigma)

    def predict(self, X):
        # 線性判別
        def g(mu, phi):
            return X @ self.invSigma @ mu - 0.5 * mu.T @ self.invSigma @ mu + np.log(phi)
        g0 = g(self.mu0, 1 - self.phi)
        g1 = g(self.mu1, self.phi)
        return (g1 > g0).astype(int)

model = GDA()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"Accuracy = {acc:.3f}")

# 決策邊界可視化
plt.figure(figsize=(4,5))
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], c='skyblue', label='Class 0')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='salmon', label='Class 1')

# meshgrid for boundary
x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
X_grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"GDA Decision Boundary (Acc={acc:.3f})")
plt.legend()
plt.show()
