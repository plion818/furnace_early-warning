import pandas as pd
import numpy as np

# === 1. 讀取資料 ===
input_csv = 'results/s1_anomaly_results.csv'
df = pd.read_csv(input_csv)

# === 2. 設定欄位名稱 ===
scaled_col = 'resistance_scaled'
raw_col = 'resistance'
resistance_series = df[raw_col].values

# === 3. 剔除標準化後絕對值大於3的樣本，計算原始電阻平均與標準差 ===
filtered_df = df[np.abs(df[scaled_col]) <= 3]
filtered_raw = filtered_df[raw_col]
mean_raw = filtered_raw.mean()

# === 4. 計算 distance_diff（每點與平均電阻的距離，單位同原始電阻）===
df['distance_diff'] = np.abs(df[raw_col] - mean_raw)


# 1. 計算完整的 slope_diff（不設 NaN）
full_slope_diff = [np.nan]
for i in range(1, len(resistance_series)):
    slope = resistance_series[i] - resistance_series[i-1]
    full_slope_diff.append(np.abs(slope))
full_slope_diff = np.array(full_slope_diff)
df['slope_diff'] = full_slope_diff

# 2. 只用斜率絕對值不超過2的部分計算 mean/std
filtered_slope_diff = full_slope_diff[(~np.isnan(full_slope_diff)) & (full_slope_diff <= 2)]
mean_slope = np.mean(filtered_slope_diff)
std_slope = np.std(filtered_slope_diff)
print(f"斜率不超過2時的 slope_diff 平均值: {mean_slope}, 標準差: {std_slope}")

# 3. slope_score 用完整 slope_diff 做標準化
df['slope_score'] = np.abs((full_slope_diff - mean_slope) / std_slope)

# # distance_diff 標準化
dist_mean = df['distance_diff'].mean()
dist_std = df['distance_diff'].std()
df['distance_score'] = np.abs((df['distance_diff'] - dist_mean) / dist_std)

# print(f"=== scores 前 10 筆 ===")
print(df[['res_spike_anomaly_score', 'distance_score', 'slope_score']].head(10))

# === 6. 匯出三個分數指標成 CSV 檔案 ===

# 讀取原始 input_csv，再將 distance_score, slope_score 新增至最後兩欄，並匯出成 scores.csv
orig_df = pd.read_csv(input_csv)
orig_df['distance_score'] = df['distance_score']
orig_df['slope_score'] = df['slope_score']
output_scores_csv = 'scores.csv'
orig_df.to_csv(output_scores_csv, index=False)
print(f"已將 distance_score、slope_score 新增至 {input_csv} 並匯出成 {output_scores_csv}")

