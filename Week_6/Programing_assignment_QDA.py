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
        self.classes = np.unique(y)
        n, d = X.shape
        self.mu = {}
        self.sigma = {}
        self.prior = {}
        for c in self.classes:
            Xc = X[y==c]
            self.mu[c] = Xc.mean(axis=0)
            self.sigma[c] = np.cov(Xc, rowvar=False)
            self.prior[c] = len(Xc)/n

    def predict(self, X):
        scores = []
        for c in self.classes:
            mu = self.mu[c]
            sigma = self.sigma[c]
            invS = np.linalg.inv(sigma)
            detS = np.linalg.det(sigma)
            term = -0.5 * np.sum((X - mu) @ invS * (X - mu), axis=1)
            term -= 0.5 * np.log(detS)
            term += np.log(self.prior[c])
            scores.append(term)
        scores = np.column_stack(scores)
        return self.classes[np.argmax(scores, axis=1)]
    
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
