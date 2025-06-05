import pandas as pd
import numpy as np

# === 規則一：檢測阻抗異常，記錄每筆資料異常命中次數與異常比例分數 ===
def detect_resistance_outliers_by_window(df, window_size, z_thresh, stride, vote_threshold):
    resistance = df['resistance'].values
    resistance_scaled = df['resistance_scaled'].values
    anomaly_counts = np.zeros(len(df), dtype=int)
    window_hits = np.zeros(len(df), dtype=int)

    for i in range(window_size, len(resistance), stride):
        # 取得當前滑動視窗的資料
        window_res = resistance[i - window_size:i]
        window_scaled = resistance_scaled[i - window_size:i]

        # Step 1: 剔除標準化後絕對值大於1的離群值
        mask = np.abs(window_scaled) < 1
        filtered_res = window_res[mask]

        if len(filtered_res) < window_size * 0.5:  # 若剩餘資料少於視窗大小的一半則跳過
            continue

        # Step 2: 根據剩餘資料計算平均與標準差
        mean_res = np.mean(filtered_res)
        std_res = np.std(filtered_res, ddof=1)

        # Step 3: 回頭檢查原本視窗的資料中哪些超出 mean ± zσ
        for j in range(i - window_size, i):
            if j >= len(resistance):
                continue
            window_hits[j] += 1
            if abs(resistance[j] - mean_res) > z_thresh * std_res:
                anomaly_counts[j] += 1

    # 計算每筆資料的異常命中率（異常次數 / 命中次數）
    anomaly_score = np.zeros(len(resistance))
    for i in range(len(resistance)):
        if window_hits[i] > 0:
            anomaly_score[i] = anomaly_counts[i] / window_hits[i]

    # 若某筆資料在其被命中的視窗中異常次數 > 指定門檻比例，則視為真正異常
    final_anomalies = [i for i in range(len(resistance)) if window_hits[i] > 0 and anomaly_score[i] >= vote_threshold]
    return final_anomalies, anomaly_score


# === 主流程 ===
# ✅ 設定檔案名稱與參數
input_csv = 'data/processed/sensorID_28_standardized.csv'
output_csv = 'results/s1_anomaly_results.csv'
window_size = 50
z_thresh = 3.0
# stride = int(window_size / 10)  # 滑動視窗步長
stride = 1
vote_threshold = 0.5  # 超過命中次數的一半即視為異常

# 讀取資料
df = pd.read_csv(input_csv)

# 只執行阻抗異常偵測
res_spike_anomalies, anomaly_score = detect_resistance_outliers_by_window(df, window_size, z_thresh, stride, vote_threshold)
df['res_spike_anomaly'] = 0
df.loc[res_spike_anomalies, 'res_spike_anomaly'] = 1

# 新增一欄異常分數：顯示每筆資料被判定為異常的比例
df['res_spike_anomaly_score'] = anomaly_score

# 輸出成 CSV 檔案
df.to_csv(output_csv, index=False)

# 顯示異常偵測摘要
print(f"✅ 結果已匯出至：{output_csv}")
print(f"（阻抗異常）異常點數量：{len(res_spike_anomalies)}")

