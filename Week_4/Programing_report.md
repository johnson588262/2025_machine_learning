# 使用多層感知器 (MLP) 進行氣象格點資料分類任務

## 1. Introduction
這次的目標是利用機器學習方法，針對氣象觀測平台所提供的**溫度格點資料**進行建模。  
原始資料以 XML 檔案提供，其中每一個格點包含經度、緯度以及對應的溫度值。由於資料中存在無效值 (以 `-999` 表示)，因此我們將問題拆解為兩個監督式學習任務：  

1. **分類任務 (Classification)**  
   - 輸入：經度、緯度  
   - 輸出：該格點的資料是否為有效值 (label = 0 或 1)  

2. **回歸任務 (Regression)**  
   - 輸入：經度、緯度  
   - 輸出：該格點的實際溫度觀測值  

上半部分是 **分類模型** 的設計與結果分析，並對模型在驗證集上的表現進行詳細評估。
下半部分才是 **回歸模型** 。

---

## 2. Methodology

### 2.1 Dataset Preparation
- **資料轉換**：  
  - 由原始 XML 檔案解析出經度、緯度及溫度值。  
  - 對於分類資料集：若溫度值為 `-999`，則標記為 label=0 (無效值)；否則為 label=1 (有效值)。  
- **特徵設計**：  
  - 使用 `longitude` (經度) 與 `latitude` (緯度) 作為輸入特徵。  
- **資料切分**：  
  - 使用 stratified split，保持正負樣本比例一致。  
  - 訓練集與驗證集比例設為 3:1。  
- **特徵標準化**：  
  - 對經度與緯度進行 zero-mean 與 unit-variance 標準化，避免數值尺度差異影響模型收斂。

---

### 2.2 Model Design
- **模型架構 (MLP)**：  
  - Input layer: 2 nodes (longitude, latitude)  
  - Hidden Layer 1: 64 nodes, activation = `tanh`  
  - Hidden Layer 2: 64 nodes, activation = `tanh`  
  - Output Layer: 1 node, activation = `sigmoid` (輸出有效值的機率)  

- **損失函數 (Loss Function)**：  
  - 採用 **Weighted Binary Cross-Entropy**  
  - 權重計算方式：  
    $$
    w_0 = \frac{N}{2 \cdot N_0}, \quad 
    w_1 = \frac{N}{2 \cdot N_1}
    $$
    其中 $N_0$ 與 $N_1$ 分別為負樣本與正樣本數量，用以處理類別不平衡問題。  

- **優化器 (Optimizer)**：  
  - Adam optimizer  
  - 參數：learning rate = $10^{-3}$，$\beta_1=0.9$，$\beta_2=0.999$，weight decay = $10^{-6}$  

- **正則化策略**：  
  - 使用 Early Stopping 機制，若驗證集 loss 在 patience 內無顯著下降則停止訓練。  

---

## 3. Experiment & Results

### 3.1 Training Process
- 訓練與驗證 Loss 曲線
![Figure_2](/Week_4/classfication%20final%20version/Figure_2.png)   
- Early stopping epoch 數: 1698。

### 3.2 Evaluation Metrics
- **Validation Accuracy**  : 0.9895
- **Validation ROC-AUC**  :0.9995

### 3.3 Visualization
- 混淆矩陣熱力圖 (Confusion Matrix Heatmap)。 
![Figure_1](/Week_4/classfication%20final%20version/Figure_1.png) 
- ROC Curve 圖表與 AUC 指標。 
![Figure_3](/Week_4/classfication%20final%20version/Figure_3.png)  

---

## 4. Discussion
- **結果解讀**：  
  - Accuracy 與 ROC-AUC 分數接近 1，代表模型對於有效/無效值的判別能力極佳。  
  - 混淆矩陣顯示誤判樣本數極少，模型能在絕大部分情況下正確分類。  
- **模型效果分析**：  
  - Weighted BCE 成功解決類別不平衡的問題，避免模型只學到「永遠預測有效值」。  
  - Bias 初始化使用 logit(prior)，幫助輸出分佈更接近實際比例，加快收斂速度。  
- **Struggle**：  
  前前後後弄到第五版本才調整好整個模型， ~~中間做出了一堆我翻硬幣都比他準的垃圾~~ Weighted BCE有效的解決了前四版模型中最常遇到的問題--全部猜1或0。 \
  前幾版大致上可以分成: \
  第一種
    - 2-hidden-layer MLP for binary classification
    - hidden: tanh, output: sigmoid
    - loss: Binary Cross Entropy (BCE)

  第二種
    Binary classification with a hand-crafted MLP on (longitude, latitude).
    - 2 hidden layers (tanh), sigmoid output
    - Weighted Binary Cross-Entropy (handles class imbalance)
    - Mini-batch SGD + L2, early stopping
    - Metrics: Accuracy, ROC-AUC; Plots: ROC curve, Confusion Matrix, Loss curves
  總而言之，這兩個最後都沒跑出甚麼好結果，可以看這個資料夾中的ver2 ver4 有當初跑完的照片
---

## 5. Conclusion
- 本研究利用多層感知器 (MLP) 成功完成氣象格點資料的分類任務。  
- 模型在驗證集上取得高準確度 (Validation Accuracy) 與接近完美的 ROC-AUC。  
- 實驗證明 **加權 BCE + Adam Optimizer + Early Stopping** 能有效解決類別不平衡與過擬合問題。  

# 回歸模型報告

## 1. Introduction
採用 **多層感知器 ( MLP)** 作為基礎架構，並引入 **Fourier features** (因為原本用的那個東西做出來真的很慘，訓練到最後都是給我吐平均值當答案，codex救我的)以捕捉經緯度與溫度之間的週期性非線性關係。  
此外，損失函數採用 **Huber Loss**，以減少異常值對模型的影響，使得模型對於極端觀測值更加穩健。



---

## 2. Methodology

### 2.1 Dataset Preparation
- **資料分割**：75% 訓練集，25% 驗證集  
- **標準化**：以訓練集計算均值與標準差，套用至訓練與驗證集  
### 2.2 Model Design
- **模型架構 (MLP+Fourier 特徵)**：  
    - Input：地理座標 (longitude, latitude) → Fourier 特徵嵌入
    - Hidden layers：2 層 fully connected，激活函數為 Tanh
    - Output：單一實數，代表預測溫度
    - 損失函數：Huber loss
    - 最佳化器：Adam
    - 正規化：輸入資料標準化

- **Early Stopping**：若驗證 loss 在一定 epoch 未改善，則提前停止訓練  
- **訓練設定**：
  - 批次大小 (batch size)：32  
  - 學習率 (lr)：1e-3  
  - 最大 epoch：2000  
  - patience：約 300  


---

## 3. Experiment & Results

### 3.1 損失曲線 (Loss Curve)
![loss_curve](/Week_4/regression%20final%20version/Figure_1.png)  
- 訓練 loss 隨 epoch 穩定下降  
- 驗證 loss 約在 **0.07 ~ 0.09 Huber** 區間震盪，整體趨勢保持平穩  
- 少量 overfitting 現象，但未導致效能大幅下降

---

### 3.2 預測結果 (y vs ŷ)
![y_vs_yhat](/Week_4/regression%20final%20version/Figure_2.png)  
- 散點圖大多分布在對角線附近，顯示模型能正確擬合溫度  
- 在極低溫 (<10°C) 與極高溫 (>28°C) 區域，仍存在較大誤差  

---

### 3.3 殘差分析 (Residuals)
![residuals](/Week_4/regression%20final%20version/Figure_3.png)  
- 殘差分布呈現近似常態，集中於 0 附近  
- 大部分預測誤差落在 ±2°C 範圍內  
- 少量長尾，顯示極端值仍較難擬合  

---

### 3.4 數值指標
- **Validation MSE**：2.8746  
- **Validation MAE**：1.1032  
- **Validation R²**：0.9250  

這代表模型平均誤差約 **1.1°C**，整體解釋變異度達 **92.5%**。

---

## 4. Discussion
1. Fourier 特徵大幅提升了模型在非線性資料上的擬合能力，相較於原始 MLP，R² 由 **-0.39 提升至 0.93**，效果顯著。  
2. 殘差分布顯示大多數樣本預測精準，但極端溫度仍有挑戰，可能需要更多特徵（如季節、海拔）。  
3. 目前模型雖有輕微 overfitting，但驗證誤差仍低，整體表現穩定，需要更多數據的驗證。

**結論**：本回歸模型已能準確預測溫度，平均誤差僅約 1°C，對應應用場景屬於可接受範圍。
