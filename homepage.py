
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px # 匯入 Plotly Express
import requests # For API calls
import io # For converting DataFrame to in-memory CSV
from anomaly_marker import AnomalyMarker
from dotenv import load_dotenv
import os

# === 頁面基礎設定 (必須是第一個 Streamlit 指令) ===
st.set_page_config(
    page_title="🔥 加熱爐數據趨勢分析儀 (API整合版)",
    page_icon="🔥",
    layout="wide"
)


# === 載入 .env 設定 ===
load_dotenv()
API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
RAW_DATA_PATH = "data/processed/sensorID_28_standardized.csv"

# === 載入主要感測資料 ===
@st.cache_data # Cache the data loading to improve performance
def load_data():
    """
    從 RAW_DATA_PATH 載入感測器資料。
    將 'record Time' 欄位轉為 datetime 物件。
    回傳:
        pd.DataFrame: 包含感測資料的 DataFrame。
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df['record Time'] = pd.to_datetime(df['record Time'])
        # Ensure required columns for the API are present, even if not used directly by Streamlit visualization always
        # This is more of a check for data integrity for the API call.
        required_cols_for_api = ['resistance', 'resistance_scaled', 'record Time']
        if not all(col in df.columns for col in required_cols_for_api):
            missing = [col for col in required_cols_for_api if col not in df.columns]
            st.error(f"資料檔 {RAW_DATA_PATH} 缺少必要欄位: {missing}。API 呼叫可能會失敗。")
            # Depending on strictness, could return None or raise error
        return df
    except FileNotFoundError:
        st.error(f"錯誤：資料檔 {RAW_DATA_PATH} 未找到。請確認檔案路徑是否正確。")
        return None
    except Exception as e:
        st.error(f"載入資料時發生錯誤: {e}")
        return None

# --- API 互動函式 ---
# Not caching this by default, as parameters might change or users might want fresh detection.
# If API calls are expensive and parameters are static for a view, @st.cache_data or @st.cache_resource could be used.
def fetch_anomalies_from_api(input_df):
    """
    使用提供的 DataFrame 呼叫異常偵測 API。
    Args:
        input_df (pd.DataFrame): 包含 'resistance', 'resistance_scaled', 'record Time' 欄位的 DataFrame。
    Returns:
        list: API 回傳的異常點列表 (字典格式)，或在錯誤時回傳 None。
    """
    if input_df is None or input_df.empty:
        st.warning("輸入 API 的資料為空，跳過偵測。")
        return []

    st.info("⏳ 正在透過 API 偵測異常點...")
    try:
        # 將 DataFrame 轉換為 CSV 字串
        csv_data = input_df.to_csv(index=False)
        files = {'file': ('data.csv', csv_data.encode('utf-8'), 'text/csv')}

        # API 參數 (可根據需求調整或從 UI 獲取)
        api_params = {
            "window_size": 200,
            "z_thresh": 3.0,
            "stride": 1,
            "vote_threshold": 0.5
        }

        headers = {}
        if API_TOKEN:
            headers["Authorization"] = f"Bearer {API_TOKEN}"
        response = requests.post(API_URL, files=files, params=api_params, headers=headers, timeout=180) # Increased timeout
        response.raise_for_status()  # 若狀態碼為 4xx 或 5xx，則拋出 HTTPError

        response_data = response.json()
        st.success(f"✅ API 異常偵測完成！共回傳 {len(response_data.get('anomalies', []))} 個潛在異常點。")
        return response_data.get('anomalies', []) # API 回傳的是包含 'anomalies' 鍵的字典

    except requests.exceptions.Timeout:
        st.error("API 請求超時。請稍後再試或檢查伺服器狀態。")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"無法連接至 API ({API_URL})。請確認 API 服務是否正在運行。")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 請求失敗 (HTTP {e.response.status_code}): {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"呼叫 API 時發生錯誤: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        st.error(f"處理 API 回應時發生非預期錯誤: {e}")
        return None



# --- 資料載入與初始化 ---
df_global = load_data() # 全域 DataFrame

# 初始化缺失標記物件
MISSING_CSV_PATH = "data/processed/sensorID_28_missing_samples.csv"
anomaly_marker = AnomalyMarker(MISSING_CSV_PATH)

# 初始化 session state
if 'time_start' not in st.session_state:
    st.session_state.time_start = datetime.strptime("2025-02-06 22:00:00", "%Y-%m-%d %H:%M:%S") if df_global is not None else datetime.now() - timedelta(hours=1)
    st.session_state.time_end = datetime.strptime("2025-02-06 23:00:00", "%Y-%m-%d %H:%M:%S") if df_global is not None else datetime.now()

if 'api_anomalies_data' not in st.session_state: # API回傳的原始異常資料
    st.session_state.api_anomalies_data = None
if 'processed_anomalies_df' not in st.session_state: # 處理並合併到主df後的異常資料 (給整體統計用)
    st.session_state.processed_anomalies_df = None


# === 頁面標題 ===
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>📈 加熱爐指標趨勢分析 (API整合)</h1>
""", unsafe_allow_html=True)

# --- 側邊欄 UI ---
st.sidebar.markdown("---")

# API 偵測按鈕
if st.sidebar.button("🔥 偵測阻抗異常點 (透過 API)"):
    if df_global is not None:
        api_result = fetch_anomalies_from_api(df_global.copy()) # Pass a copy to be safe
        if api_result is not None:
            st.session_state.api_anomalies_data = api_result
            # Process and merge for overall statistics
            anomalies_df = pd.DataFrame(st.session_state.api_anomalies_data)
            if not anomalies_df.empty:
                anomalies_df['record Time'] = pd.to_datetime(anomalies_df['record_Time']) # API回傳是record_Time
                anomalies_df.rename(columns={'score': 'api_score', 'index': 'api_original_index'}, inplace=True)

                # Merge with a copy of the global df for overall stats
                df_for_stats = df_global.copy()
                df_for_stats = pd.merge(df_for_stats,
                                        anomalies_df[['record Time', 'api_score', 'api_original_index']],
                                        on='record Time',
                                        how='left')
                df_for_stats['is_api_anomaly'] = df_for_stats['api_score'].notna()
                st.session_state.processed_anomalies_df = df_for_stats
            else: # No anomalies returned from API
                df_for_stats = df_global.copy()
                df_for_stats['is_api_anomaly'] = False
                df_for_stats['api_score'] = np.nan
                st.session_state.processed_anomalies_df = df_for_stats

        else: # API call failed
            st.session_state.api_anomalies_data = None # Clear previous results on failure
            st.session_state.processed_anomalies_df = None
    else:
        st.sidebar.error("資料尚未成功載入，無法進行 API 偵測。")


with st.sidebar.expander("⏱️ 時間選擇", expanded=True):
    st.markdown("### ⏱ 選擇資料區間")
    interval_minutes = st.selectbox("選擇時間區段長度", [15, 30, 60], index=1)

    manual_start = st.text_input("Start Time", st.session_state.time_start.strftime("%Y-%m-%d %H:%M:%S"))
    manual_end = st.text_input("End Time", st.session_state.time_end.strftime("%Y-%m-%d %H:%M:%S"))

    col_prev, col_next = st.columns(2)
    if col_prev.button("⬅ 上一段"):
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_end = st.session_state.time_start
        st.session_state.time_start = st.session_state.time_end - delta
    if col_next.button("下一段 ➡"):
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_start = st.session_state.time_end
        st.session_state.time_end = st.session_state.time_start + delta

    if st.button("🚀 生成圖表"):
        try:
            st.session_state.time_start = datetime.strptime(manual_start, "%Y-%m-%d %H:%M:%S")
            st.session_state.time_end = datetime.strptime(manual_end, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            st.error("❌ 時間格式錯誤，請使用 YYYY-MM-DD HH:MM:SS")
            st.stop()

    st.markdown(f"**目前查詢區間：**\n{st.session_state.time_start} ~ {st.session_state.time_end}")

st.sidebar.markdown("---")
with st.sidebar.expander("📊 圖表選項", expanded=True):
    columns = ['current', 'voltage', 'resistance', 'temperature']
    scaled_columns = ['current_scaled', 'voltage_scaled', 'resistance_scaled', 'temperature_scaled']
    mapping = dict(zip(columns, scaled_columns))

    selected_metrics = st.multiselect("選擇要比較的標準化指標 (1~4)", columns, default=["resistance", "temperature"])
    st.markdown("---")
    raw_option = st.selectbox("選擇要繪製的原始指標", ["none"] + columns, index=0)
    show_anomaly = st.checkbox("🔍 顯示阻抗異常點 (需先點擊偵測按鈕)")


# --- 根據時間區間過濾資料 ---
if df_global is None:
    st.warning("資料載入失敗，無法顯示圖表。請檢查資料來源。")
    st.stop()

try:
    mask = (df_global['record Time'] >= st.session_state.time_start) & (df_global['record Time'] <= st.session_state.time_end)
    df_filtered = df_global.loc[mask].copy()
except Exception as e:
    st.error(f"❌ 時間格式轉換或資料過濾時發生錯誤: {e}")
    st.stop()

# Initialize API anomaly columns in df_filtered
df_filtered['is_api_anomaly'] = False
df_filtered['api_score'] = np.nan
df_filtered['api_original_index'] = pd.NA # Using pandas NA for integer index if needed


# Merge API anomalies if available and checkbox is ticked
if show_anomaly and st.session_state.api_anomalies_data is not None:
    if st.session_state.api_anomalies_data: # Check if list is not empty
        anomalies_from_api_df = pd.DataFrame(st.session_state.api_anomalies_data)
        anomalies_from_api_df['record Time'] = pd.to_datetime(anomalies_from_api_df['record_Time'])
        anomalies_from_api_df.rename(columns={'score': 'api_score_temp', 'index': 'api_original_index_temp'}, inplace=True)

        # Merge with df_filtered
        df_filtered = pd.merge(
            df_filtered,
            anomalies_from_api_df[['record Time', 'api_score_temp', 'api_original_index_temp']],
            on='record Time',
            how='left'
        )
        # Update actual columns based on merged data
        df_filtered['is_api_anomaly'] = df_filtered['api_score_temp'].notna()
        df_filtered['api_score'] = df_filtered['api_score_temp']
        df_filtered['api_original_index'] = df_filtered['api_original_index_temp']
        # Drop temporary columns
        df_filtered.drop(columns=['api_score_temp', 'api_original_index_temp'], inplace=True)

        # 用物件導向方式標記 resistance 缺失異常
        df_filtered = anomaly_marker.mark_res_missing(df_filtered)

        num_anomalies_interval = df_filtered['is_api_anomaly'].sum()
        total_interval = len(df_filtered)
        percentage_interval = (num_anomalies_interval / total_interval) * 100 if total_interval > 0 else 0
        st.info(f"📌 API 偵測異常點數量（目前區間）：{int(num_anomalies_interval)} / {total_interval} 筆資料（{percentage_interval:.2f}%）")
    else: # API data is empty list
        st.info("API 未回傳任何異常點。")

st.success("✅ 圖表已生成。請滑鼠移動檢視原始資料詳情")

# === 標準化數據趨勢圖繪製 ===
if selected_metrics:
    metric_colors = {
        'current': 'rgba(0, 0, 255, 0.8)', 'voltage': 'rgba(0, 128, 0, 0.8)',
        'resistance': 'rgba(255, 165, 0, 0.8)', 'temperature': 'rgba(128, 0, 128, 0.8)'
    }
    fig_scaled = go.Figure()
    for metric_name in selected_metrics:
        scaled_value_format = ".4f" if metric_name == 'resistance' else ".2f"
        hover_texts_metric = []
        for i in range(len(df_filtered)):
            row = df_filtered.iloc[i]
            current_metric_scaled_val = row[mapping[metric_name]]
            hover_text = (
                f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"<b>{metric_name.capitalize()} (scaled): {current_metric_scaled_val:{scaled_value_format}}</b><br>"
                f"<b>----------- 原始數值 ---------</b><br>"
                f"Current (raw): {row['current']:.2f} A<br>"
                f"Voltage (raw): {row['voltage']:.2f} V<br>"
                f"Resistance (raw): {row['resistance']:.4f} Ω<br>"
                f"Temperature (raw): {row['temperature']:.2f} °C"
            )
            if row['is_api_anomaly'] and pd.notna(row['api_score']):
                hover_text += f"<br><b style='color:red;'>API Anomaly Score: {row['api_score']:.4f}</b>"
            hover_texts_metric.append(hover_text)

        fig_scaled.add_trace(go.Scatter(
            x=df_filtered['record Time'], y=df_filtered[mapping[metric_name]], mode='lines+markers',
            name=f'{metric_name.capitalize()} (scaled)',
            marker=dict(size=5, color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            line=dict(color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            hoverinfo='text', text=hover_texts_metric, showlegend=True
        ))

    if 'resistance' in selected_metrics and show_anomaly and df_filtered['is_api_anomaly'].any():
        anomaly_points = df_filtered[df_filtered['is_api_anomaly'] == True]
        if not anomaly_points.empty:
            # 分開標記
            res_missing_points = anomaly_points[anomaly_points['is_res_missing_anomaly'] == True]
            normal_anomaly_points = anomaly_points[anomaly_points['is_res_missing_anomaly'] == False]
            # 棕色: resistance 缺失異常
            if not res_missing_points.empty:
                res_missing_hover_texts = []
                for i in range(len(res_missing_points)):
                    row = res_missing_points.iloc[i]
                    res_missing_hover_texts.append(
                        f"<span style='color:#8B4513'><b>偵測到異常且resistance缺失 (API)</b><br>"
                        f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Resistance (scaled): {row['resistance_scaled']:.4f}<br>"
                        f"API Score: {row['api_score']:.4f}</span><br>"
                        f"<b>------------- 原始數值 ------------</b><br>"
                        f"Current: {row['current']:.2f} A<br>"
                        f"Voltage: {row['voltage']:.2f} V<br>"
                        f"Resistance: {row['resistance']:.4f} Ω<br>"
                        f"Temperature: {row['temperature']:.2f} °C"
                    )
                fig_scaled.add_trace(go.Scatter(
                    x=res_missing_points['record Time'], y=res_missing_points['resistance_scaled'],
                    mode='markers', name='Resistance Anomaly (API, 缺失)',
                    marker=dict(color='#8B4513', size=8, symbol='circle'),
                    showlegend=True, hoverinfo='text', text=res_missing_hover_texts
                ))
            # 紅色: 一般異常
            if not normal_anomaly_points.empty:
                normal_hover_texts = []
                for i in range(len(normal_anomaly_points)):
                    row = normal_anomaly_points.iloc[i]
                    normal_hover_texts.append(
                        f"<span style='color:red'><b>偵測到異常 (Resistance via API)</b><br>"
                        f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Resistance (scaled): {row['resistance_scaled']:.4f}<br>"
                        f"API Score: {row['api_score']:.4f}</span><br>"
                        f"<b>------------- 原始數值 ------------</b><br>"
                        f"Current: {row['current']:.2f} A<br>"
                        f"Voltage: {row['voltage']:.2f} V<br>"
                        f"Resistance: {row['resistance']:.4f} Ω<br>"
                        f"Temperature: {row['temperature']:.2f} °C"
                    )
                fig_scaled.add_trace(go.Scatter(
                    x=normal_anomaly_points['record Time'], y=normal_anomaly_points['resistance_scaled'],
                    mode='markers', name='Resistance Anomaly (API)',
                    marker=dict(color='red', size=8, symbol='circle'),
                    showlegend=True, hoverinfo='text', text=normal_hover_texts
                ))
    fig_scaled.update_layout(title='📊 標準化指標趨勢圖', xaxis_title='時間', yaxis_title='標準化數值', hovermode='closest', height=500, margin=dict(l=40, r=40, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.5)'))
    st.plotly_chart(fig_scaled, width="stretch")

# === 原始數據圖表繪製 ===
if raw_option != "none":
    if 'metric_colors' not in locals(): metric_colors = {'current': 'rgba(0,0,255,0.8)','voltage': 'rgba(0,128,0,0.8)','resistance': 'rgba(255,165,0,0.8)','temperature': 'rgba(128,0,128,0.8)'}
    raw_hover_texts = []
    for i in range(len(df_filtered)):
        row = df_filtered.iloc[i]
        raw_val = row[raw_option]
        raw_value_format = ".4f" if raw_option == 'resistance' else ".2f"
        hover_text = (
            f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"<b>{raw_option.capitalize()}: {raw_val:{raw_value_format}}</b><br>"
            f"Current: {row['current']:.2f} A<br>"
            f"Voltage: {row['voltage']:.2f} V<br>"
            f"Resistance: {row['resistance']:.4f} Ω<br>"
            f"Temperature: {row['temperature']:.2f} °C"
        )
        if row['is_api_anomaly'] and pd.notna(row['api_score']):
             hover_text += f"<br><b style='color:red;'>API Anomaly Score: {row['api_score']:.4f}</b>"
        raw_hover_texts.append(hover_text)

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df_filtered['record Time'], y=df_filtered[raw_option], mode='lines+markers',
        name=f'{raw_option.capitalize()} (raw)',
        marker=dict(size=5, color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        line=dict(color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        hoverinfo='text', text=raw_hover_texts, showlegend=True
    ))

    if show_anomaly and raw_option == 'resistance' and df_filtered['is_api_anomaly'].any():
        anomaly_points = df_filtered[df_filtered['is_api_anomaly'] == True]
        if not anomaly_points.empty:
            res_missing_points = anomaly_points[anomaly_points['is_res_missing_anomaly'] == True]
            normal_anomaly_points = anomaly_points[anomaly_points['is_res_missing_anomaly'] == False]
            # 棕色: resistance 缺失異常
            if not res_missing_points.empty:
                res_missing_hover_texts = []
                for i in range(len(res_missing_points)):
                    row = res_missing_points.iloc[i]
                    res_missing_hover_texts.append(
                        f"<span style='color:#8B4513'><b>偵測到異常且resistance缺失 (API)</b><br>"
                        f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Resistance (raw): {row['resistance']:.4f} Ω<br>"
                        f"API Score: {row['api_score']:.4f}</span><br>"
                        f"<b>--------- 原始數值 -------</b><br>"
                        f"Current: {row['current']:.2f} A<br>"
                        f"Voltage: {row['voltage']:.2f} V<br>"
                        f"Resistance: {row['resistance']:.4f} Ω<br>"
                        f"Temperature: {row['temperature']:.2f} °C"
                    )
                fig_raw.add_trace(go.Scatter(
                    x=res_missing_points['record Time'], y=res_missing_points['resistance'],
                    mode='markers', name='Resistance Anomaly (API, 缺失, raw)',
                    marker=dict(color='#8B4513', size=8, symbol='circle'),
                    showlegend=True, hoverinfo='text', text=res_missing_hover_texts
                ))
            # 紅色: 一般異常
            if not normal_anomaly_points.empty:
                normal_hover_texts = []
                for i in range(len(normal_anomaly_points)):
                    row = normal_anomaly_points.iloc[i]
                    normal_hover_texts.append(
                        f"<span style='color:red'><b>偵測到異常 (API)</b><br>"
                        f"Time: {row['record Time'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                        f"Resistance (raw): {row['resistance']:.4f} Ω<br>"
                        f"API Score: {row['api_score']:.4f}</span><br>"
                        f"<b>--------- 原始數值 -------</b><br>"
                        f"Current: {row['current']:.2f} A<br>"
                        f"Voltage: {row['voltage']:.2f} V<br>"
                        f"Resistance: {row['resistance']:.4f} Ω<br>"
                        f"Temperature: {row['temperature']:.2f} °C"
                    )
                fig_raw.add_trace(go.Scatter(
                    x=normal_anomaly_points['record Time'], y=normal_anomaly_points['resistance'],
                    mode='markers', name='Resistance Anomaly (API, raw)',
                    marker=dict(color='red', size=8, symbol='circle'),
                    showlegend=True, hoverinfo='text', text=normal_hover_texts
                ))
    fig_raw.update_layout(title=f'📈 原始指標趨勢圖: {raw_option.capitalize()}', xaxis_title='時間', yaxis_title='原始數值', hovermode='closest', height=500, margin=dict(l=40, r=40, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.5)'))
    st.plotly_chart(fig_raw, width="stretch")

# --- 匯出異常資料按鈕 ---
if (selected_metrics or raw_option != "none") and show_anomaly and st.session_state.api_anomalies_data is not None and df_filtered['is_api_anomaly'].any():
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    # Prepare data for download: only rows that are API anomalies within the filtered view
    export_df = df_filtered[df_filtered['is_api_anomaly'] == True].copy()
    # Select relevant columns for export, could include original index from API if needed
    export_df = export_df[['record Time', 'current', 'voltage', 'resistance', 'temperature',
                           'current_scaled', 'voltage_scaled', 'resistance_scaled', 'temperature_scaled',
                           'api_score', 'api_original_index']] # Add more as needed

    csv_data_export = export_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📤 匯出目前區間 API 偵測之異常點資料",
        data=csv_data_export,
        file_name="filtered_api_anomalies.csv",
        mime='text/csv',
        key="download-api-anomalies"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- 整體異常統計 (基於 processed_anomalies_df from session state) ---
if show_anomaly and st.session_state.processed_anomalies_df is not None:
    st.markdown("---")
    df_stats_overall = st.session_state.processed_anomalies_df

    total_all_for_info = len(df_stats_overall)
    total_anomalies_for_info = df_stats_overall['is_api_anomaly'].sum()
    percent_all_for_info = (total_anomalies_for_info / total_all_for_info) * 100 if total_all_for_info > 0 else 0
    st.info(f"📊 API 偵測異常點數量（全部資料）：{int(total_anomalies_for_info)} / {total_all_for_info} 筆資料（{percent_all_for_info:.2f}%）")

    st.markdown("---")
    st.subheader("📊 API 偵測之電阻異常比例 (整體資料)")

    anomaly_count_chart = total_anomalies_for_info
    normal_count_chart = total_all_for_info - anomaly_count_chart
    normal_percentage = (normal_count_chart / total_all_for_info) * 100 if total_all_for_info > 0 else 0
    anomaly_percentage = (anomaly_count_chart / total_all_for_info) * 100 if total_all_for_info > 0 else 0

    chart_data_stacked = [
        {'Category': 'API 電阻異常偵測', 'Segment': 'Normal', 'Count': normal_count_chart, 'Percentage': normal_percentage, 'TextOnBar': f"<b>{normal_percentage:.1f}%</b>"},
        {'Category': 'API 電阻異常偵測', 'Segment': 'Anomaly', 'Count': anomaly_count_chart, 'Percentage': anomaly_percentage, 'TextOnBar': f"<b>{anomaly_percentage:.1f}%</b>"}
    ]
    df_chart_stacked = pd.DataFrame(chart_data_stacked)

    fig_stats = px.bar(
        df_chart_stacked, x='Count', y='Category', color='Segment', orientation='h', text='TextOnBar',
        custom_data=['Segment', 'Count', 'Percentage'],
        color_discrete_map={'Normal': 'rgba(0, 128, 0, 0.7)', 'Anomaly': 'rgba(255, 0, 0, 0.7)'}
    )
    fig_stats.update_layout(
        title_text="API 電阻異常偵測：整體資料統計 (Normal vs. Anomaly Proportions)",
        xaxis_title="總數量", yaxis_title=None, height=200, showlegend=True,
        legend_title_text='類型', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_stats.update_yaxes(visible=False, showticklabels=False)
    fig_stats.update_xaxes(range=[0, total_all_for_info])
    fig_stats.update_traces(
        textposition='inside', insidetextanchor='middle', textfont_size=15, textfont_color="black",
        hovertemplate=("<b>%{customdata[0]}</b><br><br>Count: %{customdata[1]}<br>Percentage: %{customdata[2]:.1f}%<extra></extra>")
    )
    st.plotly_chart(fig_stats, width="stretch")

else:
    if show_anomaly: # Checkbox is ticked but no data in session state yet
        st.warning("請先點擊側邊欄的「偵測阻抗異常點 (透過 API)」按鈕以載入並顯示 API 偵測結果。")

st.sidebar.info("v1.1.0 - API 整合版")
