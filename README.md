# 📈 AI 股市漲跌預測系統

這是一個基於 Python 與 Streamlit 開發的互動式股市預測與分析網頁應用程式。本專案不僅使用機器學習模型來預測股價漲跌，還結合了**歷史回測**、**人機協作 (Human-in-the-loop)** 以及 **可解釋性 AI (XAI)** 的概念，讓使用者能全面掌握預測邏輯與市場動態。

---

## 🚀 快速開始 (使用 `uv`)

本專案強烈建議使用 [uv](https://github.com/astral-sh/uv) (極速的 Python 套件管理器) 來進行環境建置與執行。

### 1. 安裝 `uv` (若尚未安裝)
- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. 安裝依賴套件
如果您剛 clone 這個專案，請先讓 `uv` 同步安裝所有需要的套件：
```bash
uv sync
```
*(備註：本專案依賴 `streamlit`, `yfinance`, `pandas`, `numpy`, `scikit-learn`, `plotly`)*

### 3. 啟動系統
在終端機中執行以下指令，Streamlit 會自動在您的瀏覽器中開啟網頁介面：
```bash
uv run streamlit run prediction_K_line.py
```

---

## 💡 系統三大核心功能

### 1. 📊 純 AI 數據預測與回測
- **即時預測**：輸入股票代碼（如 `2330.TW`），系統即會抓取最新歷史資料，預測**下一個交易日**的漲跌與模型信心度。
- **歷史回測系統**：自動保留最近 20% 的交易日進行回測，模擬「隔日沖」策略，並與大盤（單純買進持有）進行累積報酬率比較。
- **牛熊市自動判定**：根據回測區間的最大回撤與報酬率，自動判定目前屬於「牛市、熊市或盤整」，協助解釋策略績效。

### 2. 🧠 貝氏權重實驗室 (Human-in-the-loop)
為了克服傳統模型難以量化突發「消息面」的缺點，本專案首創人機協作的實驗室介面：
- **AI 負責客觀技術面**：帶入 Random Forest 計算出的客觀看漲機率。
- **使用者負責主觀消息面**：透過拉桿輸入今日的市場情緒（極度悲觀 ~ 極度樂觀）。
- **貝氏機率混合**：使用者可自訂「AI vs 主觀情緒」的信任權重比例，系統即時透過精美的儀表板輸出最終的「後驗看漲機率」。

### 3. 🔍 AI 決策解釋 (Explainable AI, XAI)
打破機器學習的黑盒子！本頁面會抓出隨機森林模型的 `feature_importances_`，並繪製成直觀的長條圖。
- 清楚展示 AI 是根據哪些技術指標（如：今日成交量變化、前一日 K 線實體等）來做出明天的漲跌判斷。
- 附帶白話文的指標解釋，降低使用者的認知門檻。

---

## 🛠️ 系統架構與技術棧

- **網頁前端與 UI 框架**：`Streamlit`
- **資料獲取**：`yfinance` (Yahoo Finance API)
- **資料處理與特徵工程**：`pandas`, `numpy`
  - *包含：日報酬率、K線實體比例、上下影線比例、成交量變化率，以及時間序列 Lag (平移) 特徵。*
- **機器學習演算法**：`scikit-learn` (`RandomForestClassifier`)
- **互動式圖表視覺化**：`plotly.graph_objects` (K線圖、績效折線圖、長條圖、儀表板)
