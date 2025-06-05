import pandas as pd

# 讀取資料
file_path = "data\\processed\\sensorID_28_standardized.csv"
df = pd.read_csv(file_path)

# 確保 Record Time 資料是 datetime 格式
df['record Time'] = pd.to_datetime(df['record Time'])

# 保留未過濾的資料
original_df = df.copy()
original_df['HourBin'] = original_df['record Time'].dt.floor('H')
original_counts = original_df['HourBin'].value_counts().sort_index()
original_summary = original_counts.reset_index()
original_summary.columns = ['record Time', 'Original_Count']

# 移除 resistance_scaled 絕對值 >= 3 的離群樣本
filtered_df = df[df['resistance_scaled'].abs() < 3].copy()
filtered_df['HourBin'] = filtered_df['record Time'].dt.floor('H')
filtered_counts = filtered_df['HourBin'].value_counts().sort_index()
filtered_summary = filtered_counts.reset_index()
filtered_summary.columns = ['record Time', 'Filtered_Count']

# 合併兩者結果便於對比

# 以 'record Time' 欄位合併
comparison = pd.merge(original_summary, filtered_summary, on='record Time', how='outer').sort_values('record Time')
comparison = comparison.fillna(0)
comparison[['Original_Count', 'Filtered_Count']] = comparison[['Original_Count', 'Filtered_Count']].astype(int)

# 每小時計算 filtered_df 的平均值
hourly_avg = filtered_df.groupby('HourBin')[['current', 'voltage', 'resistance', 'temperature']].mean().reset_index()
hourly_avg.columns = ['record Time', 'Avg_Current', 'Avg_Voltage', 'Avg_Resistance', 'Avg_Temperature']

# ⭐ 新增：標準化平均值欄位
avg_cols = ['Avg_Current', 'Avg_Voltage', 'Avg_Resistance', 'Avg_Temperature']
for col in avg_cols:
    mean = hourly_avg[col].mean()
    std = hourly_avg[col].std()
    hourly_avg[col + '_Scaled'] = (hourly_avg[col] - mean) / std

# 合併平均值到 comparison 表中
comparison = pd.merge(comparison, hourly_avg, on='record Time', how='left')

# # 預覽輸出
# print(comparison.head())

# 找出 Original_Count 與 Filtered_Count 差異最大的前五筆
comparison['Count_Diff'] = (comparison['Original_Count'] - comparison['Filtered_Count']).abs()
top5_diff = comparison.sort_values('Count_Diff', ascending=False).head(10)
print('Original_Count 與 Filtered_Count 差異最大的前五筆：')
print(top5_diff[['record Time', 'Original_Count', 'Filtered_Count', 'Count_Diff']])

# 匯出 comparison 結果成 CSV
# comparison.to_csv('outlier_remove.csv', index=False)
# print('已匯出 comparison 結果至 outlier_remove.csv')
