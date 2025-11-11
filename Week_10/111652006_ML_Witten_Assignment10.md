# Written assignment 
已知正向 SDE：

$$
dx_t = f(x_t,t)\,dt + g(x_t,t)\,dW_t.
$$

其機率密度 $p(x,t)$ 滿足 Fokker–Planck（Kolmogorov forward）方程：

$$
\frac{\partial}{\partial t} p(x,t)
= -\frac{\partial}{\partial x}\big(f(x,t)\,p(x,t)\big)
+ \frac{1}{2}\frac{\partial^2}{\partial x^2}\big(g^2(x,t)\,p(x,t)\big).
$$

另一方面，若存在一條 **probability flow ODE**：

$$
dx_t = \tilde f(x_t,t)\,dt,
$$

其誘導的密度演化必滿足continuity equation：

$$
\frac{\partial}{\partial t}p(x,t)
= -\frac{\partial}{\partial x}\big(\tilde f(x,t)\,p(x,t)\big).
$$

令兩個密度演化相同，則：

$$
\partial_x \big(\tilde f(x,t)\,p(x,t)\big) = 
\partial_x \big( f(x,t)\,p(x,t)-\frac{1}{2}\frac{\partial}{\partial x}\big(g^2(x,t)\,p(x,t)\big) \big).
$$

雙邊同時積分，多出的常數只跟時間有關，可以算至 $\tilde f(x,t)$ 中，

$$
\tilde f(x,t)\,p(x,t)
= f(x,t)\,p(x,t)
-\frac{1}{2}\frac{\partial}{\partial x}\big(g^2(x,t)\,p(x,t)\big).
$$

將右側展開並除以 $p(x,t)>0$：

$$
\tilde f(x,t)
= f(x,t)
-\frac{1}{2}\frac{\partial}{\partial x}g^2(x,t)
-\frac{g^2(x,t)}{2}\,\frac{\partial_x p(x,t)}{p(x,t)}.
$$

注意 $\partial_x p / p = \partial_x \log p$ ，故

$$
\tilde f(x,t)
= f(x,t)
-\frac{1}{2}\frac{\partial}{\partial x}g^2(x,t)
-\frac{g^2(x,t)}{2}\,\frac{\partial}{\partial x}\log p(x,t).
$$

因此對應的 probability flow ODE 為

$$
dx_t =
\Big[
f(x_t,t)
-\frac{1}{2}\frac{\partial}{\partial x}g^2(x_t,t)
-\frac{g^2(x_t,t)}{2}\,\frac{\partial}{\partial x}\log p(x_t,t)
\Big]\,dt.
$$

多維且非各向同性還沒想到怎麼寫0.0 各項同性就
在 $x\in\mathbb{R}^d$ 時，Fokker–Planck 為

$$
\partial_t p
= -\nabla\!\cdot\!(f\,p)
+ \tfrac{1}{2}\sum_{i,j}\partial_{x_i x_j}\!\big((GG^\top)_{ij}\,p\big),
$$

若 $G(x,t)=g(x,t)I$ ，則

$$
\dot x = \tilde f(x,t)
= f(x,t) - \tfrac{1}{2}\nabla g^2(x,t) - \tfrac{1}{2}g^2(x,t)\,\nabla \log p(x,t).
$$

# RoboTaxi的號角與全無人車時代的來臨 

## 一、前言

我認為在未來五年內，RoboTaxi（自動駕駛計程車） 將慢慢普及至全球城市，逐漸取代原來的計程車系統，更甚者隨著人工智慧與感測技術的進步，二十年內，人類駕駛與個人持有車輛動產概念將逐漸退出舞台，取而代之的是以深度學習與強化學習為核心的自動駕駛系統。
我相信，當所有車輛全面自動化後，「車禍」將成為歷史名詞，而人類的出行方式也將被徹底重塑。

## 二、技術的核心基礎

我認為RoboTaxi 的實現主要依賴 強化學習（Reinforcement Learning） 與 監督式學習（Supervised Learning） 的結合。分析如下:

**監督式學習** 用於訓練模型的「感知系統」，讓車輛能夠正確辨識影像中的車輛、行人、交通號誌、道路標線等物件。這需要大量標註過的影像資料（例如 Waymo、Tesla 收集的行車影片）。

**強化學習** 則用於「決策與控制」層，AI 透過模擬或真實環境中的試誤過程學習最佳行車策略。例如如何在不同交通情境下轉彎、煞車或閃避障礙物。這讓 AI 能根據即時環境回饋自我調整行為。此外，強化學習當中還有一個子領域在這題目下扮演著很重要的角色-- **多智能體學習（Multi-Agent Learning）**，每輛自駕車不僅自己學習，還能將經驗同步至雲端，與全球其他車輛共享。這代表整個車隊會隨時間一起進化，形成真正的「群體智慧（Swarm Intelligence）」。

**資料來源**：車輛感測器收集的多模態資料（攝影機影像、雷射雷達、GPS、速度與方向資訊）、其他智能體提供的雲端資料。

**目標訊號（Target Signal）**：

- 在監督學習中，目標是正確的物件標籤或行為示範（如人類駕駛的轉向與加速操作）。

- 在強化學習中，目標是「最大化獎勵（reward）」——例如安全、平順且高效率的行車路徑。

此外，RoboTaxi 存在學習回饋與環境互動。在模擬器中，AI 可以不斷測試不同的駕駛策略並從環境得到回饋（如碰撞懲罰、到達目的地獎勵）。在真實世界中，模型也會持續從感測數據與實際路況中學習，進行「持續學習（Continual Learning）」與「聯邦學習（Federated Learning）」更新，以提升系統安全與穩定性。

## 三、小結
會想寫這個主題是因為最近看到Elon Musk講的那句話「開車在未來只會成為小眾人的愛好」(“Any cars that are being made that don’t have full autonomy will have negative value. It will be like owning a horse. You will only be owning it for sentimental reasons.”)

根據世界衛生組織統計，全球每年約有 130 萬人死於交通事故，其中 超過九成是人為錯誤造成的。
RoboTaxi 的出現，意味著所有駕駛決策都交由 AI 處理。
AI 具備遠超人類的感知與反應速度，這種「集體智能」的駕駛方式，使得每一輛車不再是獨立個體，而是整個城市交通系統的神經元，車禍一詞將逐漸離開我們的世界，接踵而來的則是全新的法規與責任、資料隱私與安全性問題，儘管如此我還是認為全面無人車會是人類對安全、智慧與永續的共同追求。