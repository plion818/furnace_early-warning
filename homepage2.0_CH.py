import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px # 匯入 Plotly Express

# === 頁面基礎設定 (必須是第一個 Streamlit 指令) ===
st.set_page_config(
    page_title="🔥 加熱爐數據趨勢分析儀",
    page_icon="🔥",
    layout="wide"
)


# === 載入主要感測資料（直接從異常檢測結果） ===
@st.cache_data
def load_data():
    """
    直接從 'results/s1_anomaly_results.csv' 載入所有所需資料。
    將 'record Time' 欄位轉為 datetime 物件。
    回傳:
        pd.DataFrame: 包含所有感測與異常資料的 DataFrame。
    """
    df = pd.read_csv("results/s1_anomaly_results.csv")
    df['record Time'] = pd.to_datetime(df['record Time'])
    return df

# --- 資料載入與初始化 ---
df = load_data()  # 直接載入所有資料
anomaly_df = df  # 直接指向同一份資料，方便後續程式碼相容

# === 頁面標題 ===
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>📈 加熱爐指標趨勢分析</h1>
""", unsafe_allow_html=True)

# --- 側邊欄 UI ---
st.sidebar.markdown("---") # 視覺分隔線

# 初始化 session state 的時間區間（如尚未設定）
if 'time_start' not in st.session_state:
    st.session_state.time_start = datetime.strptime("2025-02-06 22:00:00", "%Y-%m-%d %H:%M:%S")
    st.session_state.time_end = datetime.strptime("2025-02-06 23:00:00", "%Y-%m-%d %H:%M:%S")

# 側邊欄展開區塊：時間選擇
with st.sidebar.expander("⏱️ 時間選擇", expanded=True): # 時間選擇 UI
    st.markdown("### ⏱ 選擇資料區間")
    interval_minutes = st.selectbox("選擇時間區段長度", [15, 30, 60], index=1) # 時間區段長度

    # 手動輸入時間
    manual_start = st.text_input("Start Time", st.session_state.time_start.strftime("%Y-%m-%d %H:%M:%S"))
    manual_end = st.text_input("End Time", st.session_state.time_end.strftime("%Y-%m-%d %H:%M:%S"))

    # 時間切換按鈕
    col_prev, col_next = st.columns(2)
    if col_prev.button("⬅ 上一段"): # 上一區段
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_end = st.session_state.time_start
        st.session_state.time_start = st.session_state.time_end - delta
    if col_next.button("下一段 ➡"): # 下一區段
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_start = st.session_state.time_end
        st.session_state.time_end = st.session_state.time_start + delta

    # 按下按鈕時更新時間區間
    if st.button("🚀 生成圖表"): # 產生圖表按鈕
        try:
            st.session_state.time_start = datetime.strptime(manual_start, "%Y-%m-%d %H:%M:%S")
            st.session_state.time_end = datetime.strptime(manual_end, "%Y-%m-%d %H:%M:%S")
        except ValueError: # 捕捉時間格式錯誤
            st.error("❌ 時間格式錯誤，請使用 YYYY-MM-DD HH:MM:SS")
            st.stop() # 格式錯誤時停止執行

    # 顯示目前查詢區間
    st.markdown(f"**目前查詢區間：**\n{st.session_state.time_start} ~ {st.session_state.time_end}")

# 側邊欄展開區塊：圖表選項
st.sidebar.markdown("---") # 視覺分隔線
with st.sidebar.expander("📊 圖表選項", expanded=True): # 圖表與異常顯示選項
    columns = ['current', 'voltage', 'resistance', 'temperature'] # 可選指標
    scaled_columns = ['current_scaled', 'voltage_scaled', 'resistance_scaled', 'temperature_scaled'] # 對應標準化欄位
    mapping = dict(zip(columns, scaled_columns)) # 原始欄位對應標準化欄位

    # 選擇要比較的標準化指標
    selected_metrics = st.multiselect("選擇要比較的標準化指標 (1~4)", columns, default=["resistance", "temperature"])

    # 選擇要繪製的原始資料指標
    st.markdown("---") # 視覺分隔線
    raw_option = st.selectbox("選擇要繪製的原始指標", ["none"] + columns, index=0)

    # 勾選是否顯示異常點
    show_anomaly = st.checkbox("🔍 顯示阻抗異常點")


# --- 根據時間區間過濾資料 ---
try:
    # 建立遮罩，過濾出選定時間區間的資料
    mask = (df['record Time'] >= st.session_state.time_start) & (df['record Time'] <= st.session_state.time_end)
    df_filtered = df.loc[mask].copy() # 套用遮罩並複製過濾後的資料
except Exception as e: # 捕捉時間轉換或過濾錯誤
    st.error(f"❌ 時間格式轉換或資料過濾時發生錯誤: {e}")
    st.stop() # 若過濾失敗則停止執行

st.success("✅ 圖表已生成。請滑鼠移動檢視原始資料詳情")

# 若使用者想看異常點但異常檢測檔案不存在則警告
if show_anomaly and anomaly_df is None:
    st.warning("⚠️ 異常檢測結果文件 (results/s1_anomaly_results.csv) 未找到或讀取失敗。異常點相關功能將不可用。")


# 若勾選顯示異常點且異常資料存在，則計算異常統計
if show_anomaly and anomaly_df is not None:
    # 計算過濾區間內的異常統計
    if 'res_spike_anomaly' in df.columns:
        df_filtered['res_spike_anomaly'] = df.loc[df_filtered.index, 'res_spike_anomaly']
        df_filtered['res_spike_anomaly_score'] = df.loc[df_filtered.index, 'res_spike_anomaly_score']
        total = len(df_filtered)
        num_anomalies = df_filtered['res_spike_anomaly'].sum()
        percentage = (num_anomalies / total) * 100 if total > 0 else 0
        st.info(f"📌 異常點數量（區間）：{int(num_anomalies)} / {total} 筆資料（{percentage:.2f}%）")

        total_all = len(df)
        total_anomalies_all = df['res_spike_anomaly'].sum()
        percent_all = (total_anomalies_all / total_all) * 100 if total_all > 0 else 0
    else:
        percent_all = 0


# === 標準化數據趨勢圖繪製 ===
# 此區段負責繪製所選標準化指標的趨勢圖
if selected_metrics:
    metric_colors = { # 定義各指標顏色
        'current': 'rgba(0, 0, 255, 0.8)',      # 藍色
        'voltage': 'rgba(0, 128, 0, 0.8)',      # 綠色
        'resistance': 'rgba(255, 165, 0, 0.8)', # 橘色
        'temperature': 'rgba(128, 0, 128, 0.8)' # 紫色
    }

    fig_scaled = go.Figure() # 初始化圖表

    # 逐一將每個選定指標加入圖表
    for metric_name in selected_metrics:
        scaled_value_format = ".4f" if metric_name == 'resistance' else ".2f"

        # 建立每個資料點的詳細 hover 訊息
        hover_texts_metric = []
        for i, t in enumerate(df_filtered['record Time']):
            current_metric_scaled_val = df_filtered[mapping[metric_name]].iloc[i]
            # 取得原始值
            cur = df_filtered['current'].iloc[i]
            vol = df_filtered['voltage'].iloc[i]
            res = df_filtered['resistance'].iloc[i]
            temp = df_filtered['temperature'].iloc[i]

            current_hover_text = (
                f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"<b>{metric_name.capitalize()} (scaled): {current_metric_scaled_val:{scaled_value_format}}</b><br>"
                f"<b>----------- 原始數值 ---------</b><br>"
                f"Current (raw): {cur:.2f} A<br>"
                f"Voltage (raw): {vol:.2f} V<br>"
                f"Resistance (raw): {res:.4f} Ω<br>"
                f"Temperature (raw): {temp:.2f} °C"
            )
            # 若有異常分數且非 NaN 則加上
            if 'res_spike_anomaly_score' in df_filtered.columns:
                score = df_filtered['res_spike_anomaly_score'].iloc[i]
                if pd.notna(score):
                    current_hover_text += f"<br>Anomaly Score: {score:.4f}"
            hover_texts_metric.append(current_hover_text)

        fig_scaled.add_trace(go.Scatter(
            x=df_filtered['record Time'],
            y=df_filtered[mapping[metric_name]],
            mode='lines+markers',
            name=f'{metric_name.capitalize()} (scaled)', # 圖例名稱
            marker=dict(size=5, color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            line=dict(color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            hoverinfo='text',
            text=hover_texts_metric,
            showlegend=True # 顯示圖例
        ))

    # 若選擇 resistance 且顯示異常點，則加上異常標記
    anomaly_data_loaded = anomaly_df is not None
    if 'resistance' in selected_metrics and show_anomaly and anomaly_data_loaded and 'res_spike_anomaly' in df_filtered.columns:
        anomaly_points = df_filtered[df_filtered['res_spike_anomaly'] == 1]
        if not anomaly_points.empty:
            anomaly_hover_texts = []
            for i, t_anomaly in enumerate(anomaly_points['record Time']):
                cur_anomaly = anomaly_points['current'].iloc[i]
                vol_anomaly = anomaly_points['voltage'].iloc[i]
                res_anomaly_raw = anomaly_points['resistance'].iloc[i] # 原始電阻
                temp_anomaly = anomaly_points['temperature'].iloc[i]
                score_anomaly = anomaly_points['res_spike_anomaly_score'].iloc[i]
                scaled_res_anomaly = anomaly_points['resistance_scaled'].iloc[i]

                anomaly_hover_texts.append(
                    f"<span style='color:red'><b>偵測到異常 (Resistance)</b><br>"
                    f"Time: {t_anomaly.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Resistance (scaled): {scaled_res_anomaly:.4f}<br>"
                    f"Score: {score_anomaly:.4f}</span><br>"
                    f"<b>------------- 原始數值 ------------</b><br>"
                    f"Current: {cur_anomaly:.2f} A<br>"
                    f"Voltage: {vol_anomaly:.2f} V<br>"
                    f"Resistance: {res_anomaly_raw:.4f} Ω<br>"
                    f"Temperature: {temp_anomaly:.2f} °C"
                )

            fig_scaled.add_trace(go.Scatter(
                x=anomaly_points['record Time'],
                y=anomaly_points['resistance_scaled'], # 標記異常點
                mode='markers',
                name='Resistance Anomaly',
                marker=dict(color='red', size=8, symbol='circle'),
                showlegend=True,
                hoverinfo='text',
                text=anomaly_hover_texts
            ))

    # 設定圖表版面
    fig_scaled.update_layout(
        title='📊 標準化指標趨勢圖',
        xaxis_title='時間',
        yaxis_title='標準化數值',
        hovermode='closest', # 滑鼠移到最近點顯示
        height=500, # 圖表高度
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )
    st.plotly_chart(fig_scaled, use_container_width=True) # 顯示圖表

# === 原始數據圖表繪製 ===
# 此區段負責繪製所選原始指標的圖表
if raw_option != "none": # 若有選擇原始指標
    # 若 metric_colors 尚未定義則定義
    if 'metric_colors' not in locals():
        metric_colors = {
            'current': 'rgba(0, 0, 255, 0.8)',
            'voltage': 'rgba(0, 128, 0, 0.8)',
            'resistance': 'rgba(255, 165, 0, 0.8)',
            'temperature': 'rgba(128, 0, 128, 0.8)'
        }

    # 建立每個資料點的詳細 hover 訊息
    raw_hover_texts = []
    for i, t in enumerate(df_filtered['record Time']):
        cur = df_filtered['current'].iloc[i]
        vol = df_filtered['voltage'].iloc[i]
        res = df_filtered['resistance'].iloc[i]
        temp = df_filtered['temperature'].iloc[i]
        raw_val = df_filtered[raw_option].iloc[i] # 選定指標的原始值

        raw_value_format = ".4f" if raw_option == 'resistance' else ".2f" # 格式化

        current_hover_text = (
            f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"<b>{raw_option.capitalize()}: {raw_val:{raw_value_format}}</b><br>"
            f"Current: {cur:.2f} A<br>"
            f"Voltage: {vol:.2f} V<br>"
            f"Resistance: {res:.4f} Ω<br>"
            f"Temperature: {temp:.2f} °C"
        )
        # 若有異常分數且非 NaN 則加上
        if 'res_spike_anomaly_score' in df_filtered.columns:
            score = df_filtered['res_spike_anomaly_score'].iloc[i]
            if pd.notna(score):
                current_hover_text += f"<br>Anomaly Score: {score:.4f}"
        raw_hover_texts.append(current_hover_text)

    fig_raw = go.Figure() # 建立原始資料圖表
    # 加入主線條
    fig_raw.add_trace(go.Scatter(
        x=df_filtered['record Time'],
        y=df_filtered[raw_option], # Y 軸為原始資料
        mode='lines+markers',
        name=f'{raw_option.capitalize()} (raw)',
        marker=dict(size=5, color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        line=dict(color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        hoverinfo='text',
        text=raw_hover_texts,
        showlegend=True # 顯示圖例
    ))

    # 若選擇 resistance 且顯示異常點，則加上異常標記
    if show_anomaly and raw_option == 'resistance' and anomaly_df is not None and 'res_spike_anomaly' in df_filtered.columns:
        anomaly_points = df_filtered[df_filtered['res_spike_anomaly'] == 1] # 過濾異常點
        if not anomaly_points.empty:
            # 建立異常點 hover 訊息
            raw_anomaly_hover_texts = []
            for i, t_anomaly in enumerate(anomaly_points['record Time']):
                cur_anomaly = anomaly_points['current'].iloc[i]
                vol_anomaly = anomaly_points['voltage'].iloc[i]
                res_anomaly = anomaly_points['resistance'].iloc[i] # 原始電阻
                temp_anomaly = anomaly_points['temperature'].iloc[i]
                score_anomaly = anomaly_points['res_spike_anomaly_score'].iloc[i]
                raw_anomaly_hover_texts.append(
                    f"<span style='color:red'><b>偵測到異常</b><br>"
                    f"Time: {t_anomaly.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Resistance (raw): {res_anomaly:.4f} Ω<br>"
                    f"Score: {score_anomaly:.4f}</span><br>"
                    f"<b>--------- 原始數值 -------</b><br>"
                    f"Current: {cur_anomaly:.2f} A<br>"
                    f"Voltage: {vol_anomaly:.2f} V<br>"
                    f"Resistance: {res_anomaly:.4f} Ω<br>"
                    f"Temperature: {temp_anomaly:.2f} °C"
                )
            # 加入異常點標記
            fig_raw.add_trace(go.Scatter(
                x=anomaly_points['record Time'],
                y=anomaly_points['resistance'], # Y 軸為原始電阻
                mode='markers',
                name='Resistance Anomaly (raw)',
                marker=dict(color='red', size=8, symbol='circle'),
                showlegend=True, # 顯示圖例
                hoverinfo='text',
                text=raw_anomaly_hover_texts
            ))

    # 設定原始資料圖表版面
    fig_raw.update_layout(
        title=f'📈 原始指標趨勢圖: {raw_option.capitalize()}', # 標題
        xaxis_title='時間',
        yaxis_title='原始數值',
        hovermode='closest',
        height=500, # 圖表高度
        margin=dict(l=40, r=40, t=60, b=40), # 邊界
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.5)') # 圖例樣式
    )
    st.plotly_chart(fig_raw, use_container_width=True) # 顯示圖表


# --- 匯出異常資料按鈕 ---
# 若有顯示圖表且顯示異常點且異常資料存在，則顯示下載按鈕
if (selected_metrics or raw_option != "none") and show_anomaly and anomaly_df is not None and 'res_spike_anomaly' in df_filtered.columns:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True) # 右對齊
    # 準備下載資料（過濾後的異常點）
    csv_data = df_filtered[df_filtered['res_spike_anomaly'] == 1].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📤 匯出目前區間異常點資料", # 按鈕標籤
        data=csv_data,
        file_name="filtered_anomalies.csv", # 預設檔名
        mime='text/csv',
        key="download-below-plot"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --- 整體異常統計 ---
# 若有勾選顯示異常點且異常資料存在，則顯示整體異常統計
if show_anomaly and anomaly_df is not None and 'res_spike_anomaly' in df.columns: # 檢查主資料表有異常欄位
    st.markdown("---") # 視覺分隔線
    # 再次計算整體異常統計或使用前面已計算的 percent_all
    if 'total_anomalies_all' not in locals() or 'total_all' not in locals() or \
       not ('res_spike_anomaly' in df.columns and hasattr(df, 'res_spike_anomaly')): # 確認欄位存在
        # 若前面變數不存在則重新計算
        if 'res_spike_anomaly' in df.columns:
            total_all_for_info = len(df)
            total_anomalies_for_info = df['res_spike_anomaly'].sum()
        else:
            total_all_for_info = len(df) # 即使沒異常欄位也顯示總數
            total_anomalies_for_info = 0 # 無異常
    else: # 若前面已計算則直接用
        total_all_for_info = total_all
        total_anomalies_for_info = total_anomalies_all

    percent_all_for_info = (total_anomalies_for_info / total_all_for_info) * 100 if total_all_for_info > 0 else 0
    st.info(f"📊 異常點數量（全部 - 電阻偵測）：{int(total_anomalies_for_info)} / {total_all_for_info} 筆資料（{percent_all_for_info:.2f}%）")

    # --- 電阻異常統計堆疊橫條圖 ---
    st.markdown("---")
    st.subheader("📊 電阻異常偵測比例 (整體資料)") # 子標題

    # 堆疊橫條圖資料
    anomaly_count_chart = df['res_spike_anomaly'].sum()
    total_points_chart = len(df)
    normal_count_chart = total_points_chart - anomaly_count_chart

    normal_percentage = (normal_count_chart / total_points_chart) * 100 if total_points_chart > 0 else 0
    anomaly_percentage = (anomaly_count_chart / total_points_chart) * 100 if total_points_chart > 0 else 0

    chart_data_stacked = [
        {
            'Category': '電阻異常偵測',
            'Segment': 'Normal',
            'Count': normal_count_chart,
            'Percentage': normal_percentage,
            'TextOnBar': f"<b>{normal_percentage:.1f}%</b>"
        },
        {
            'Category': '電阻異常偵測',
            'Segment': 'Anomaly',
            'Count': anomaly_count_chart,
            'Percentage': anomaly_percentage,
            'TextOnBar': f"<b>{anomaly_percentage:.1f}%</b>"
        }
    ]
    df_chart_stacked = pd.DataFrame(chart_data_stacked)

    # 使用 Plotly Express 繪製堆疊橫條圖
    fig_stats = px.bar(
        df_chart_stacked,
        x='Count',
        y='Category',
        color='Segment',
        orientation='h',
        text='TextOnBar', # 條上顯示文字
        custom_data=['Segment', 'Count', 'Percentage'], # hover 顯示資料
        color_discrete_map={
            'Normal': 'rgba(0, 128, 0, 0.7)',  # 綠色
            'Anomaly': 'rgba(255, 0, 0, 0.7)' # 紅色
        }
    )

    # 設定堆疊橫條圖版面
    fig_stats.update_layout(
        title_text="電阻異常偵測：整體資料統計 (Normal vs. Anomaly Proportions)",
        xaxis_title="總數量",
        yaxis_title=None, # 隱藏 y 軸標題
        height=200, # 單條高度
        showlegend=True,
        legend_title_text='類型',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_stats.update_yaxes(visible=False, showticklabels=False) # 隱藏 y 軸
    fig_stats.update_xaxes(range=[0, total_points_chart]) # x 軸範圍

    # 設定條上文字與 hover
    fig_stats.update_traces(
        textposition='inside',
        insidetextanchor='middle',
        textfont_size=15, # 字體大小
        textfont_color="black", # 字體顏色
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>" + # 類型
            "Count: %{customdata[1]}<br>" +      # 數量
            "Percentage: %{customdata[2]:.1f}%" + # 百分比
            "<extra></extra>" # 隱藏 trace info
        )
    )

    st.plotly_chart(fig_stats, use_container_width=True) # 顯示圖表
