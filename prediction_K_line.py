import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 網頁 UI 設定 ---
st.set_page_config(page_title="AI 股市預測系統", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 股市漲跌預測系統")

# --- 側邊欄 ---
st.sidebar.header("🔍 股票設定")
stock_code = st.sidebar.text_input("股票代碼", value="2330.TW")
training_period = st.sidebar.select_slider("訓練數據年限", options=[1, 2, 3, 5], value=2)

st.sidebar.divider()
app_mode = st.sidebar.radio("選擇模式", ["📊 純 AI 數據預測", "🧠 貝氏權重實驗室", "🔍 AI 決策解釋 (XAI)"])

# --- 核心邏輯：修正資料處理 ---
def feature_engineering(df):
    # 關鍵修正：確保移除 yfinance 的多重索引格式
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 強制轉換為 Series 避免 DataFrame 衝突
    close_ser = df['Close'].squeeze()
    open_ser = df['Open'].squeeze()
    high_ser = df['High'].squeeze()
    low_ser = df['Low'].squeeze()
    vol_ser = df['Volume'].squeeze()

    # 計算特徵
    new_df = pd.DataFrame(index=df.index)
    new_df['Daily_Return'] = close_ser.pct_change()
    new_df['Body_Size'] = (close_ser - open_ser) / open_ser
    new_df['Vol_Change'] = vol_ser.pct_change()
    
    # 計算上下影線 (使用 squeeze 確保是 1D 資料)
    max_oc = pd.concat([open_ser, close_ser], axis=1).max(axis=1)
    min_oc = pd.concat([open_ser, close_ser], axis=1).min(axis=1)
    
    new_df['Upper_Shadow'] = (high_ser - max_oc) / open_ser
    new_df['Lower_Shadow'] = (min_oc - low_ser) / open_ser
    
    # 建立包含今日與前 4 天的特徵 (共 5 天)
    feature_cols = []
    for i in range(0, 5):
        new_df[f'Lag_{i}_Ret'] = new_df['Daily_Return'].shift(i)
        new_df[f'Lag_{i}_Body'] = new_df['Body_Size'].shift(i)
        new_df[f'Lag_{i}_Vol'] = new_df['Vol_Change'].shift(i)
        feature_cols.extend([f'Lag_{i}_Ret', f'Lag_{i}_Body', f'Lag_{i}_Vol'])
    
    # 預測目標：明天的日報酬率是否大於 0
    # 注意：使用 shift(-1) 會讓最後一天 (今日) 的 Target 變成 NaN
    new_df['Target'] = (new_df['Daily_Return'].shift(-1) > 0).astype(int)
    new_df.loc[new_df.index[-1], 'Target'] = np.nan # 確保最後一天是 NaN，因為明天還沒發生
    
    # 處理可能出現的無限大數值 (例如成交量為0導致的 pct_change 無限大)
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 移除特徵有缺漏的資料 (前面的天數)，保留最後一天用來預測明天
    df_clean = new_df.dropna(subset=feature_cols)
    return df_clean, feature_cols, df # 回傳處理後的資料, 特徵名, 以及原始資料用於畫圖

# --- 執行流程 ---
if stock_code:
    with st.spinner('正在從網路抓取最新數據並分析...'):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=training_period * 365)
        raw_data = yf.download(stock_code, start=start_date, end=end_date)
        
        try:
            ticker = yf.Ticker(stock_code)
            stock_name = ticker.info.get('shortName', stock_code)
        except:
            stock_name = stock_code
            
        st.sidebar.info(f"📌 股票名稱: {stock_name}")
        
        if not raw_data.empty and len(raw_data) > 50:
            df_model, feature_cols, clean_raw = feature_engineering(raw_data)
            
            # --- 歷史回測 (Backtesting) ---
            # 取出有完整特徵與目標值的資料 (排除最後一天)
            historical_data = df_model.dropna(subset=['Target'])
            X_hist = historical_data[feature_cols]
            y_hist = historical_data['Target']
            
            # 切割 80% 訓練集, 20% 測試集(回測)
            split_idx = int(len(X_hist) * 0.8)
            X_train, X_test = X_hist.iloc[:split_idx], X_hist.iloc[split_idx:]
            y_train, y_test = y_hist.iloc[:split_idx], y_hist.iloc[split_idx:]
            
            model_bt = RandomForestClassifier(n_estimators=100, random_state=42)
            model_bt.fit(X_train, y_train)
            
            y_pred = model_bt.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            # 計算策略報酬：預測明天會漲(1)，則賺取明天的日報酬率 (即 df_model['Daily_Return'].shift(-1))
            test_returns = df_model['Daily_Return'].shift(-1).loc[X_test.index]
            test_returns = test_returns.fillna(0) # 防呆
            
            strategy_returns = y_pred * test_returns
            cum_strategy = (1 + strategy_returns).cumprod() - 1
            cum_buy_hold = (1 + test_returns).cumprod() - 1
            
            # --- 牛/熊市自動判定 ---
            if not cum_buy_hold.empty:
                bh_final_ret = cum_buy_hold.iloc[-1]
                bh_max = cum_buy_hold.max()
                # 簡單判定邏輯
                if bh_final_ret > 0.20:
                    market_status = "🟢 牛市"
                elif (bh_max - bh_final_ret) / (1 + bh_max) > 0.20 or bh_final_ret < -0.20:
                    market_status = "🔴 熊市"
                else:
                    market_status = "🟡 盤整"
            else:
                market_status = "未知"

            # --- 最終模型 (用 100% 歷史資料預測真正的明天) ---
            model_full = RandomForestClassifier(n_estimators=100, random_state=42)
            model_full.fit(X_hist, y_hist)
            
            # 預測明天
            current_X = df_model[feature_cols].tail(1)
            prediction = model_full.predict(current_X)[0]
            prob = model_full.predict_proba(current_X)[0]

            if app_mode == "📊 純 AI 數據預測":
                # --- UI 顯示 ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    res = "🟢 看漲" if prediction == 1 else "🔴 看跌"
                    st.metric("AI 預測結果", res)
                with col2:
                    st.metric("模型信心度", f"{max(prob):.1%}")
                with col3:
                    last_p = clean_raw['Close'].iloc[-1]
                    st.metric("最新收盤價", f"{float(last_p):.2f}")

                # K 線圖
                st.divider()
                last_30 = clean_raw.tail(30)
                fig = go.Figure(data=[go.Candlestick(
                    x=last_30.index,
                    open=last_30['Open'].squeeze(), 
                    high=last_30['High'].squeeze(),
                    low=last_30['Low'].squeeze(), 
                    close=last_30['Close'].squeeze()
                )])
                fig.update_layout(title=f"【{stock_name}】最近 30 日走勢 (互動式 K 線)", xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig, width='stretch')

                # 回測結果圖表
                st.divider()
                st.subheader(f"📊 歷史回測績效 (最近 20% 交易日) - {market_status}")
                
                bt_col1, bt_col2, bt_col3 = st.columns(3)
                with bt_col1:
                    st.metric("模型回測準確率", f"{accuracy:.1%}")
                with bt_col2:
                    strat_ret = cum_strategy.iloc[-1] if not cum_strategy.empty else 0
                    st.metric("AI 策略累積報酬", f"{strat_ret:.2%}")
                with bt_col3:
                    bh_ret = cum_buy_hold.iloc[-1] if not cum_buy_hold.empty else 0
                    st.metric("單純持有累積報酬", f"{bh_ret:.2%}")

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=cum_strategy.index, y=cum_strategy, mode='lines', name='AI 隔日沖策略'))
                fig_bt.add_trace(go.Scatter(x=cum_buy_hold.index, y=cum_buy_hold, mode='lines', name='單純買進持有', line=dict(dash='dot')))
                fig_bt.update_layout(title="AI 策略 vs 單純持有 累積報酬率比較", hovermode="x unified")
                st.plotly_chart(fig_bt, width='stretch')

            elif app_mode == "🧠 貝氏權重實驗室":
                st.header("🧠 貝氏權重實驗室 (Human-in-the-loop)")
                st.markdown("結合 **AI 客觀技術面** 與 **您對市場消息的主觀判斷 (先驗機率)**，得出最終的後驗看漲機率。")
                
                ai_prob = prob[1] * 100 if len(prob) > 1 else (100 if prediction == 1 else 0) # 看漲機率 (0~100)
                
                st.divider()
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("1. 市場訊號設定")
                    st.info(f"🤖 **AI 技術面客觀看漲機率**：{ai_prob:.1f}%")
                    
                    st.caption("請根據您的觀察輸入以下面向情緒 (-100 ~ +100)。\n**若不確定或無法評估，請直接保留 0 即可！**")
                    news_sentiment = st.slider("🗣️ 消息面情緒", min_value=-100, max_value=100, value=0)
                    chip_sentiment = st.slider("💰 籌碼面情緒", min_value=-100, max_value=100, value=0)
                    fund_sentiment = st.slider("🏢 基本面情緒", min_value=-100, max_value=100, value=0)
                    
                    # 只有當拉桿不為 0 時，才將其納入平均計算 (忽略未評估的項目)
                    active_sentiments = [s for s in (news_sentiment, chip_sentiment, fund_sentiment) if s != 0]
                    if len(active_sentiments) > 0:
                        avg_sentiment = sum(active_sentiments) / len(active_sentiments)
                    else:
                        avg_sentiment = 0  # 全部都沒拉，就是中立 0
                        
                    # 將 -100 ~ 100 映射為 0% ~ 100% 機率
                    human_prob = (avg_sentiment + 100) / 2
                    st.write(f"👉 綜合換算主觀看漲先驗機率：**{human_prob:.1f}%**")
                    
                with col_right:
                    st.subheader("2. 權重分配 (貝氏混合)")
                    ai_weight = st.slider("🤖 AI 技術面 信任權重 (%)", min_value=0, max_value=100, value=50)
                    human_weight = 100 - ai_weight
                    st.write(f"🗣️ 您的主觀情緒總權重將為：**{human_weight}%**")
                    
                # 計算最終機率
                final_prob = (ai_prob * (ai_weight / 100)) + (human_prob * (human_weight / 100))
                
                st.divider()
                st.subheader("🎯 最終後驗預測結果")
                
                # 繪製一個簡單的儀表板 (Gauge) - 改為指針樣式
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = final_prob,
                    number = {"suffix": "%", "valueformat": ".1f"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "綜合看漲機率"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"}, # 隱藏原本的深藍色進度條
                        'steps': [
                            {'range': [0, 40], 'color': "lightcoral"},
                            {'range': [40, 60], 'color': "lightyellow"},
                            {'range': [60, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 6}, # 以黑色粗線充當指針
                            'thickness': 1, # 貫穿整個儀表板
                            'value': final_prob
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, width='stretch')
                
                if final_prob > 50:
                    st.success(f"綜合評估結果：建議 **🟢 看漲** (機率大於 50%)")
                else:
                    st.error(f"綜合評估結果：建議 **🔴 看跌** (機率小於 50%)")
                    
            elif app_mode == "🔍 AI 決策解釋 (XAI)":
                st.header("🔍 AI 決策解釋性 (Explainable AI)")
                st.markdown("本頁面展示 AI 模型 (Random Forest) 在預測時，最依賴的**技術特徵重要性權重**。這有助於打開黑盒子，了解 AI 的決策邏輯。")
                
                importances = model_full.feature_importances_
                imp_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                feature_names_zh = {
                    'Lag_0_Ret': '今日報酬率', 'Lag_0_Body': '今日 K線實體', 'Lag_0_Vol': '今日成交量變化',
                    'Lag_1_Ret': '前1日報酬率', 'Lag_1_Body': '前1日 K線實體', 'Lag_1_Vol': '前1日成交量變化',
                    'Lag_2_Ret': '前2日報酬率', 'Lag_2_Body': '前2日 K線實體', 'Lag_2_Vol': '前2日成交量變化',
                    'Lag_3_Ret': '前3日報酬率', 'Lag_3_Body': '前3日 K線實體', 'Lag_3_Vol': '前3日成交量變化',
                    'Lag_4_Ret': '前4日報酬率', 'Lag_4_Body': '前4日 K線實體', 'Lag_4_Vol': '前4日成交量變化',
                }
                readable_features = [f"{feature_names_zh.get(f, f)} ({f})" for f in imp_df['Feature']]
                
                fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'],
                    y=readable_features,
                    orientation='h',
                    marker_color='royalblue'
                ))
                fig_imp.update_layout(title="特徵重要性權重 (Feature Importances)", height=600, xaxis_title="重要性權重", yaxis_title="特徵指標")
                st.plotly_chart(fig_imp, width='stretch')
                
                st.info("""
                💡 **指標名詞白話文解釋：**
                * **報酬率**：當天的收盤價比前一天漲或跌了多少。這代表股票近期的「**價格動能與趨勢**」。
                * **K線實體 (Body Size)**：收盤價與開盤價的差距比例。實體越長，代表當天買方或賣方的「**力量越懸殊、表態越明確**」。
                * **成交量變化**：今天的交易量比昨天多或少。爆量通常代表「**市場情緒熱烈或有大戶進場**」，常常是行情轉折的訊號。
                
                👉 **如何看懂這張圖？**
                圖表中越靠**上方**（長度越長）的長條，代表 AI 認為這個指標對預測明天的漲跌**越重要**！
                例如：如果「今日成交量變化」排在最上面，代表 AI 發現近期的買賣熱度是決定這支股票漲跌的最大關鍵。
                """)
        else:
            st.error("無法抓取資料，請檢查股票代碼（台股需加 .TW）")

