# piecewise_sklearn.py
# 需求: numpy pandas matplotlib scikit-learn
# 讀取:
#   - temperature_classification.csv  (columns: longitude, latitude, label)
#   - temperature_regression.csv     (columns: longitude, latitude, <target>)
# 產物:
#   - combined_predictions.csv
#   - qda_boundary.png, regression_scatter.png, h_grid.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

CLS_CSV = "temperature_classification.csv"
REG_CSV = "temperature_regression.csv"

# ---------- 讀檔 ----------
def ensure_path(csv_path: str) -> Path:
    p = Path(csv_path)
    if not p.is_file():
        p = Path(__file__).parent / p.name
    if not p.is_file():
        raise FileNotFoundError(f"找不到資料檔：{p.resolve()}")
    print(f"[INFO] 讀取資料：{p.resolve()}")
    return p

path1 = ensure_path(CLS_CSV)
path2 = ensure_path(REG_CSV)
df_cls = pd.read_csv(path1)
df_reg = pd.read_csv(path2)

X_cls = df_cls[["longitude", "latitude"]].to_numpy(dtype=float)
y_cls = df_cls["label"].to_numpy()

candidate_targets = [c for c in df_reg.columns
                     if c.lower() in ["temp","temperature","target","y","label","value"]]
target_col = candidate_targets[0] if candidate_targets else df_reg.columns[-1]
X_reg = df_reg[["longitude", "latitude"]].to_numpy(dtype=float)
y_reg = df_reg[target_col].to_numpy(dtype=float)

# ---------- 訓練 / 測試分割 ----------
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cls, y_cls, test_size=0.30, random_state=42, stratify=y_cls)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.30, random_state=42)

# ---------- 模型 ----------
clf = QDA(reg_param=1e-6)        # C(x)
reg = LinearRegression()         # R(x)

clf.fit(Xc_tr, yc_tr)
reg.fit(Xr_tr, yr_tr)

# 評估
acc = clf.score(Xc_te, yc_te)
yr_pred = reg.predict(Xr_te)
rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
#rmse = mean_squared_error(yr_te, yr_pred, squared=False) sklearn版本太舊跑不了這個 以防萬一手動開根
mae  = mean_absolute_error(yr_te, yr_pred)
r2   = r2_score(yr_te, yr_pred)

print(f"[Classifier C(x)] QDA Accuracy: {acc:.3f}")
print(f"[Regressor R(x)] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

# ---------- 組合函數 h(x) ----------
def h_predict(X):
    c = clf.predict(X)
    r = reg.predict(X)
    return np.where(c == 1, r, -999.0)

# 在回歸資料上生成對照表
C_x = clf.predict(X_reg)
R_x = reg.predict(X_reg)
h_x = np.where(C_x == 1, R_x, -999.0)

out = pd.DataFrame({
    "longitude": df_reg["longitude"],
    "latitude":  df_reg["latitude"],
    "C_x":       C_x,
    "R_x":       R_x,
    "h_x":       h_x
})
out.to_csv("combined_predictions.csv", index=False)
print("Saved: combined_predictions.csv")

# ---------- 視覺化 ----------
# 1) QDA 決策邊界
x_min, x_max = X_cls[:,0].min()-0.1, X_cls[:,0].max()+0.1
y_min, y_max = X_cls[:,1].min()-0.1, X_cls[:,1].max()+0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Zc = clf.predict(grid).reshape(xx.shape)

plt.figure(figsize=(5, 8))
plt.contourf(xx, yy, Zc, alpha=0.25)
plt.scatter(X_cls[y_cls==0,0], X_cls[y_cls==0,1], s=6, label="Class 0")
plt.scatter(X_cls[y_cls==1,0], X_cls[y_cls==1,1], s=6, label="Class 1")
plt.title(f"QDA Decision Boundary (Acc={acc:.3f})")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout()
plt.savefig("qda_boundary.png", dpi=160)

# 2) 回歸 真值 vs. 預測
plt.figure(figsize=(5, 8))
plt.scatter(yr_te, yr_pred, s=10)
mn, mx = float(min(yr_te.min(), yr_pred.min())), float(max(yr_te.max(), yr_pred.max()))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True"); plt.ylabel("Predicted")
plt.title(f"Linear Regression (RMSE={rmse:.3f}, R2={r2:.3f})")
plt.tight_layout()
plt.savefig("regression_scatter.png", dpi=160)

# 3) h(x) 網格熱圖
Zr = reg.predict(grid).reshape(xx.shape)
Zh = Zr.copy()
Zh[Zc != 1] = -999.0

plt.figure(figsize=(5, 8))
im = plt.imshow(Zh, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal")
plt.colorbar(im, label="h(x) value", shrink=0.7)
plt.scatter(X_reg[:,0], X_reg[:,1], s=4)
plt.title("Combined Piecewise Model h(x)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("h_grid.png", dpi=160)

print("Saved: qda_boundary.png, regression_scatter.png, h_grid.png")
