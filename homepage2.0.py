import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px # Import Plotly Express

# === 頁面基礎設定 (Must be the first Streamlit command) ===
st.set_page_config(
    page_title="🔥 加熱爐數據趨勢分析儀",
    page_icon="🔥",
    layout="wide"
)

# === 載入主要感測資料 ===
@st.cache_data
def load_data():
    """
    Loads the main sensor data from the preprocessed CSV file.
    Converts the 'record Time' column to datetime objects.

    Returns:
        pd.DataFrame: DataFrame containing the sensor data.
    """
    df = pd.read_csv("data/processed/sensorID_28_standardized.csv")
    df['record Time'] = pd.to_datetime(df['record Time'])
    return df

# === 載入異常檢測結果（包含異常標記與分數） ===
@st.cache_data
def load_anomaly_results():
    """
    Loads anomaly detection results from 'results/anomaly_results.csv'.

    Returns:
        pd.DataFrame or None: DataFrame with anomaly data if successful,
                              None if the file is not found or cannot be read.
    """
    try:
        return pd.read_csv("results/anomaly_results.csv")
    except:
        return None

# --- Data Loading and Initialization ---
df = load_data()  # Load main sensor data
anomaly_df = load_anomaly_results()  # Load anomaly detection results (if available)

# === 頁面標題 ===
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>📈 加熱爐指標趨勢分析</h1>
""", unsafe_allow_html=True)

# --- Sidebar UI ---
st.sidebar.markdown("---") # Visual separator

# Initialize session state for time range if not already set
if 'time_start' not in st.session_state:
    st.session_state.time_start = datetime.strptime("2025-02-06 22:00:00", "%Y-%m-%d %H:%M:%S")
    st.session_state.time_end = datetime.strptime("2025-02-06 23:00:00", "%Y-%m-%d %H:%M:%S")

# Sidebar Expander: Time Selection
with st.sidebar.expander("⏱️ 時間選擇", expanded=True): # Time selection UI
    st.markdown("### ⏱ 選擇資料區間")
    interval_minutes = st.selectbox("選擇時間區段長度", [15, 30, 60], index=1) # Time interval length

    # Manual time input
    manual_start = st.text_input("Start Time", st.session_state.time_start.strftime("%Y-%m-%d %H:%M:%S"))
    manual_end = st.text_input("End Time", st.session_state.time_end.strftime("%Y-%m-%d %H:%M:%S"))

    # Time navigation buttons
    col_prev, col_next = st.columns(2)
    if col_prev.button("⬅ 上一段"): # Previous interval
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_end = st.session_state.time_start
        st.session_state.time_start = st.session_state.time_end - delta
    if col_next.button("下一段 ➡"): # Next interval
        delta = timedelta(minutes=interval_minutes)
        st.session_state.time_start = st.session_state.time_end
        st.session_state.time_end = st.session_state.time_start + delta

    # Update time range on button click
    if st.button("🚀 生成圖表"): # Generate chart button
        try:
            st.session_state.time_start = datetime.strptime(manual_start, "%Y-%m-%d %H:%M:%S")
            st.session_state.time_end = datetime.strptime(manual_end, "%Y-%m-%d %H:%M:%S")
        except ValueError: # Catch parsing error for invalid date format
            st.error("❌ 時間格式錯誤，請使用 YYYY-MM-DD HH:MM:SS")
            st.stop() # Stop execution if time format is incorrect

    # Display current time range
    st.markdown(f"**目前查詢區間：**\n{st.session_state.time_start} ~ {st.session_state.time_end}")

# Sidebar Expander: Chart Options
st.sidebar.markdown("---") # Visual separator
with st.sidebar.expander("📊 圖表選項", expanded=True): # Chart and anomaly display options
    columns = ['current', 'voltage', 'resistance', 'temperature'] # Available metrics
    scaled_columns = ['current_scaled', 'voltage_scaled', 'resistance_scaled', 'temperature_scaled'] # Corresponding scaled column names
    mapping = dict(zip(columns, scaled_columns)) # Map original to scaled column names

    # Metric selection for scaled data comparison
    selected_metrics = st.multiselect("Select Scaled Metrics to Compare (1~4)", columns, default=["resistance", "temperature"])

    # Option to plot raw data for a single metric
    st.markdown("---") # Visual separator
    raw_option = st.selectbox("Select Raw Metric to Plot", ["none"] + columns, index=0)

    # Checkbox to show/hide anomaly points
    show_anomaly = st.checkbox("🔍 顯示阻抗異常點")


# --- Data Filtering based on Time Selection ---
try:
    # Create a mask to filter data within the selected time range
    mask = (df['record Time'] >= st.session_state.time_start) & (df['record Time'] <= st.session_state.time_end)
    df_filtered = df.loc[mask].copy() # Apply mask and create a copy for filtered data
except Exception as e: # Catch potential errors during time conversion or filtering
    st.error(f"❌ 時間格式轉換或資料過濾時發生錯誤: {e}")
    st.stop() # Stop execution if filtering fails

st.success("✅ 圖表已生成。請滑鼠移動檢視原始資料詳情")

# Warn if user wants to see anomalies but the anomaly results file is missing
if show_anomaly and anomaly_df is None:
    st.warning("⚠️ 異常檢測結果文件 (results/anomaly_results.csv) 未找到或讀取失敗。異常點相關功能將不可用。")

# Apply anomaly flags/scores if 'show_anomaly' is checked and data is available
if show_anomaly and anomaly_df is not None:
    # Merge anomaly data into the main DataFrame (df) first
    # This assumes 'record Time' can be used for merging or that indices align
    # For simplicity, this example assumes anomaly_df might have a subset of indices from df
    # or specific columns to be joined. If anomaly_df is separate, a merge might be needed:
    # df = pd.merge(df, anomaly_df[['record Time', 'res_spike_anomaly', 'res_spike_anomaly_score']], on='record Time', how='left')
    # The current implementation assumes anomaly_df columns are directly assignable,
    # which implies anomaly_df might have been processed to align with df.
    # For robustness, ensure indices align or use a proper merge.
    # The original code directly assigns, so we keep that structure but add comments.

    # Assigning pre-loaded anomaly columns to the main dataframe
    # Ensure these columns exist in anomaly_df and are correctly aligned with df
    if 'res_spike_anomaly' in anomaly_df.columns:
        df['res_spike_anomaly'] = anomaly_df['res_spike_anomaly']
    if 'res_spike_anomaly_score' in anomaly_df.columns:
        df['res_spike_anomaly_score'] = anomaly_df['res_spike_anomaly_score']

    # Apply these anomaly columns to the time-filtered DataFrame
    if 'res_spike_anomaly' in df.columns:
        df_filtered['res_spike_anomaly'] = df.loc[df_filtered.index, 'res_spike_anomaly']
    if 'res_spike_anomaly_score' in df.columns:
        df_filtered['res_spike_anomaly_score'] = df.loc[df_filtered.index, 'res_spike_anomaly_score']

    # Calculate anomaly statistics for the filtered range
    if 'res_spike_anomaly' in df_filtered.columns:
        total = len(df_filtered)
        num_anomalies = df_filtered['res_spike_anomaly'].sum()
        percentage = (num_anomalies / total) * 100 if total > 0 else 0
        st.info(f"📌 異常點數量（區間）：{int(num_anomalies)} / {total} 筆資料（{percentage:.2f}%）")

    # Calculate overall anomaly statistics for the entire dataset (if anomaly data was loaded)
    if 'res_spike_anomaly' in df.columns: # Check if column exists in main df
        total_all = len(df)
        total_anomalies_all = df['res_spike_anomaly'].sum()
        percent_all = (total_anomalies_all / total_all) * 100 if total_all > 0 else 0
        # This overall stat will be displayed later, after charts. We store it here.
        # The variable percent_all will be used.
    else: # If anomaly column wasn't even in the main df (e.g. anomaly_df was empty or malformed)
        percent_all = 0 # Default to 0 if not calculable


# === 標準化數據趨勢圖繪製 ===
# This section handles plotting for selected scaled metrics.
if selected_metrics:
    metric_colors = { # Define consistent colors for metrics
        'current': 'rgba(0, 0, 255, 0.8)',      # Blue
        'voltage': 'rgba(0, 128, 0, 0.8)',      # Green
        'resistance': 'rgba(255, 165, 0, 0.8)', # Orange
        'temperature': 'rgba(128, 0, 128, 0.8)' # Purple
    }

    fig_scaled = go.Figure() # Initialize a single figure for all scaled metrics

    # Iterate through each selected metric to add its trace to the combined chart
    for metric_name in selected_metrics:
        scaled_value_format = ".4f" if metric_name == 'resistance' else ".2f"

        # Construct detailed hover text for each data point of this specific metric
        hover_texts_metric = []
        for i, t in enumerate(df_filtered['record Time']):
            current_metric_scaled_val = df_filtered[mapping[metric_name]].iloc[i]
            # Raw values from df_filtered (which is a time-filtered copy of original df)
            cur = df_filtered['current'].iloc[i]
            vol = df_filtered['voltage'].iloc[i]
            res = df_filtered['resistance'].iloc[i]
            temp = df_filtered['temperature'].iloc[i]

            current_hover_text = (
                f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"<b>{metric_name.capitalize()} (scaled): {current_metric_scaled_val:{scaled_value_format}}</b><br>"
                #f"<hr>"
                f"<b>----------- Raw Values ---------</b><br>"
                f"Current (raw): {cur:.2f} A<br>"
                f"Voltage (raw): {vol:.2f} V<br>"
                f"Resistance (raw): {res:.4f} Ω<br>"
                f"Temperature (raw): {temp:.2f} °C"
            )
            # Add anomaly score if it exists and is not NaN
            if 'res_spike_anomaly_score' in df_filtered.columns:
                score = df_filtered['res_spike_anomaly_score'].iloc[i]
                if pd.notna(score):
                    current_hover_text += f"<br>Anomaly Score: {score:.4f}"
            hover_texts_metric.append(current_hover_text)

        fig_scaled.add_trace(go.Scatter(
            x=df_filtered['record Time'],
            y=df_filtered[mapping[metric_name]],
            mode='lines+markers',
            name=f'{metric_name.capitalize()} (scaled)', # Name for the legend
            marker=dict(size=5, color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            line=dict(color=metric_colors.get(metric_name, 'rgba(0,0,0,0.7)')),
            hoverinfo='text',
            text=hover_texts_metric,
            showlegend=True # Show legend for each metric line
        ))

    # Add anomaly markers for 'resistance' if selected and enabled
    anomaly_data_loaded = anomaly_df is not None
    if 'resistance' in selected_metrics and show_anomaly and anomaly_data_loaded and 'res_spike_anomaly' in df_filtered.columns:
        anomaly_points = df_filtered[df_filtered['res_spike_anomaly'] == 1]
        if not anomaly_points.empty:
            anomaly_hover_texts = []
            for i, t_anomaly in enumerate(anomaly_points['record Time']):
                cur_anomaly = anomaly_points['current'].iloc[i]
                vol_anomaly = anomaly_points['voltage'].iloc[i]
                res_anomaly_raw = anomaly_points['resistance'].iloc[i] # Raw resistance
                temp_anomaly = anomaly_points['temperature'].iloc[i]
                score_anomaly = anomaly_points['res_spike_anomaly_score'].iloc[i]
                scaled_res_anomaly = anomaly_points['resistance_scaled'].iloc[i]

                anomaly_hover_texts.append(
                    f"<span style='color:red'><b>Anomaly Detected (Resistance)</b><br>"
                    f"Time: {t_anomaly.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Resistance (scaled): {scaled_res_anomaly:.4f}<br>"
                    f"Score: {score_anomaly:.4f}</span><br>" # Span closed
                    #f"<hr>"
                    f"<b>------------- Raw Values ------------</b><br>"
                    f"Current: {cur_anomaly:.2f} A<br>"
                    f"Voltage: {vol_anomaly:.2f} V<br>"
                    f"Resistance: {res_anomaly_raw:.4f} Ω<br>"
                    f"Temperature: {temp_anomaly:.2f} °C"
                )

            fig_scaled.add_trace(go.Scatter(
                x=anomaly_points['record Time'],
                y=anomaly_points['resistance_scaled'], # Y-value for anomaly markers on scaled chart
                mode='markers',
                name='Resistance Anomaly',
                marker=dict(color='red', size=8, symbol='circle'),
                showlegend=True,
                hoverinfo='text',
                text=anomaly_hover_texts
            ))

    # Configure layout for the combined scaled chart
    fig_scaled.update_layout(
        title='📊 Combined Scaled Metrics Trend',
        xaxis_title='Time',
        yaxis_title='Scaled Value',
        hovermode='closest', # Show hover for the closest point
        height=500, # Adjusted height for combined chart
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
    st.plotly_chart(fig_scaled, use_container_width=True) # Display the combined chart

# === 原始數據圖表繪製 ===
# This section handles plotting for the selected raw metric.
if raw_option != "none": # Check if a raw metric is selected
    # Define metric_colors if not already defined (e.g., if scaled_metrics was empty)
    if 'metric_colors' not in locals():
        metric_colors = {
            'current': 'rgba(0, 0, 255, 0.8)',
            'voltage': 'rgba(0, 128, 0, 0.8)',
            'resistance': 'rgba(255, 165, 0, 0.8)',
            'temperature': 'rgba(128, 0, 128, 0.8)'
        }

    # Construct detailed hover text for each data point in the raw chart
    raw_hover_texts = []
    for i, t in enumerate(df_filtered['record Time']):
        cur = df_filtered['current'].iloc[i]
        vol = df_filtered['voltage'].iloc[i]
        res = df_filtered['resistance'].iloc[i]
        temp = df_filtered['temperature'].iloc[i]
        raw_val = df_filtered[raw_option].iloc[i] # Value of the selected raw metric

        raw_value_format = ".4f" if raw_option == 'resistance' else ".2f" # Formatting for the primary raw value

        current_hover_text = (
            f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"<b>{raw_option.capitalize()}: {raw_val:{raw_value_format}}</b><br>" # Prominent display of selected raw metric
            #f"<hr>"
            f"Current: {cur:.2f} A<br>"
            f"Voltage: {vol:.2f} V<br>"
            f"Resistance: {res:.4f} Ω<br>"
            f"Temperature: {temp:.2f} °C"
        )
        # Add anomaly score if it exists and is not NaN
        if 'res_spike_anomaly_score' in df_filtered.columns:
            score = df_filtered['res_spike_anomaly_score'].iloc[i]
            if pd.notna(score):
                current_hover_text += f"<br>Anomaly Score: {score:.4f}"
        raw_hover_texts.append(current_hover_text)

    fig_raw = go.Figure() # Create figure for raw data plot
    # Add main trace for the raw metric
    fig_raw.add_trace(go.Scatter(
        x=df_filtered['record Time'],
        y=df_filtered[raw_option], # Y-values are the raw data of the selected metric
        mode='lines+markers',
        name=f'{raw_option.capitalize()} (raw)',
        marker=dict(size=5, color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        line=dict(color=metric_colors.get(raw_option, 'rgba(0,0,0,0.7)')),
        hoverinfo='text',
        text=raw_hover_texts,
        showlegend=True # Show legend for the raw metric line
    ))

    # Add anomaly markers if requested, applicable (resistance), and data is available
    if show_anomaly and raw_option == 'resistance' and anomaly_df is not None and 'res_spike_anomaly' in df_filtered.columns:
        anomaly_points = df_filtered[df_filtered['res_spike_anomaly'] == 1] # Filter for anomaly points
        if not anomaly_points.empty:
            # Construct hover text for raw anomaly points
            raw_anomaly_hover_texts = []
            for i, t_anomaly in enumerate(anomaly_points['record Time']):
                cur_anomaly = anomaly_points['current'].iloc[i]
                vol_anomaly = anomaly_points['voltage'].iloc[i]
                res_anomaly = anomaly_points['resistance'].iloc[i] # This is the raw resistance value for anomaly
                temp_anomaly = anomaly_points['temperature'].iloc[i]
                score_anomaly = anomaly_points['res_spike_anomaly_score'].iloc[i]
                raw_anomaly_hover_texts.append(
                    f"<span style='color:red'><b>Anomaly Detected</b><br>"
                    f"Time: {t_anomaly.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Resistance (raw): {res_anomaly:.4f} Ω<br>"
                    f"Score: {score_anomaly:.4f}</span><br>" # Span closed
                    #f"<hr>"
                    f"<b>--------- Raw Values -------</b><br>"
                    f"Current: {cur_anomaly:.2f} A<br>"
                    f"Voltage: {vol_anomaly:.2f} V<br>"
                    f"Resistance: {res_anomaly:.4f} Ω<br>" # Raw resistance repeated for clarity
                    f"Temperature: {temp_anomaly:.2f} °C"
                )
            # Add trace for raw anomaly markers
            fig_raw.add_trace(go.Scatter(
                x=anomaly_points['record Time'],
                y=anomaly_points['resistance'], # Y-value is raw resistance for anomaly
                mode='markers',
                name='Resistance Anomaly (raw)',
                marker=dict(color='red', size=8, symbol='circle'),
                showlegend=True, # Show legend for raw anomalies
                hoverinfo='text',
                text=raw_anomaly_hover_texts
            ))

    # Configure layout for the raw data figure
    fig_raw.update_layout(
        title=f'📈 Raw Metric Trend: {raw_option.capitalize()}', # Title for raw chart
        xaxis_title='Time',
        yaxis_title='Raw Value',
        hovermode='closest',
            height=500, # Height of the raw chart
            margin=dict(l=40, r=40, t=60, b=40), # Margins
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.5)') # Legend styling
    )
    st.plotly_chart(fig_raw, use_container_width=True) # Display raw chart


# --- Download Button for Anomaly Data ---
# Display download button if any chart is visible and anomalies are shown and anomaly data exists
if (selected_metrics or raw_option != "none") and show_anomaly and anomaly_df is not None and 'res_spike_anomaly' in df_filtered.columns:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True) # Align button to the right
    # Prepare data for download (filtered anomalies)
    csv_data = df_filtered[df_filtered['res_spike_anomaly'] == 1].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📤 匯出目前區間異常點資料", # Button label
        data=csv_data,
        file_name="filtered_anomalies.csv", # Default file name
        mime='text/csv',
        key="download-below-plot"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --- Overall Anomaly Statistics ---
# Display overall anomaly statistics if requested and data is available
if show_anomaly and anomaly_df is not None and 'res_spike_anomaly' in df.columns: # Check if res_spike_anomaly was successfully added to df
    st.markdown("---") # Visual separator
    # Calculate overall anomaly stats again here or use stored 'percent_all' if sure about its scope and state
    # For clarity, recalculating or ensuring total_anomalies_all and total_all are correctly scoped
    # These variables are used for the st.info message below
    if 'total_anomalies_all' not in locals() or 'total_all' not in locals() or \
       not ('res_spike_anomaly' in df.columns and hasattr(df, 'res_spike_anomaly')): # Ensure df has the column
        # Fallback or recalculation if variables from anomaly processing block aren't set or column is missing
        if 'res_spike_anomaly' in df.columns:
            total_all_for_info = len(df)
            total_anomalies_for_info = df['res_spike_anomaly'].sum()
        else:
            total_all_for_info = len(df) # Total points even if no anomaly info
            total_anomalies_for_info = 0 # No anomalies if column is missing
    else: # Use pre-calculated values if available from the anomaly processing block
        total_all_for_info = total_all
        total_anomalies_for_info = total_anomalies_all

    percent_all_for_info = (total_anomalies_for_info / total_all_for_info) * 100 if total_all_for_info > 0 else 0
    st.info(f"📊 異常點數量（全部 - 電阻偵測）：{int(total_anomalies_for_info)} / {total_all_for_info} 筆資料（{percent_all_for_info:.2f}%）")

    # --- Stacked Horizontal Bar Chart for Resistance Anomaly Statistics (Overall Data) ---
    st.markdown("---")
    st.subheader("📊 電阻異常偵測比例 (整體資料)") # Updated subheader

    # Data for the stacked bar chart
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

    # Create the stacked horizontal bar chart using Plotly Express
    fig_stats = px.bar(
        df_chart_stacked,
        x='Count',
        y='Category',
        color='Segment',
        orientation='h',
        text='TextOnBar', # Use the new field for on-bar text
        custom_data=['Segment', 'Count', 'Percentage'], # Data for hovertemplate
        color_discrete_map={
            'Normal': 'rgba(0, 128, 0, 0.7)',  # Green
            'Anomaly': 'rgba(255, 0, 0, 0.7)' # Red
        }
    )

    # Update layout for the stacked bar chart
    fig_stats.update_layout(
        title_text="電阻異常偵測：整體資料統計 (Normal vs. Anomaly Proportions)",
        xaxis_title="總數量 (Total Count)",
        yaxis_title=None, # Hide y-axis title
        height=200, # Adjusted height for a single bar
        showlegend=True,
        legend_title_text='類型 (Segment Type)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_stats.update_yaxes(visible=False, showticklabels=False) # Hide y-axis line and labels
    fig_stats.update_xaxes(range=[0, total_points_chart]) # Ensure x-axis covers the total range

    # Update trace configuration for text display and hover
    fig_stats.update_traces(
        textposition='inside',
        insidetextanchor='middle',
        textfont_size=15, # Updated font size
        textfont_color="black", # Added font color
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>" + # Segment Name
            "Count: %{customdata[1]}<br>" +      # Count
            "Percentage: %{customdata[2]:.1f}%" + # Percentage
            "<extra></extra>" # Hide trace info
        )
    )

    st.plotly_chart(fig_stats, use_container_width=True) # Display the chart