
import pandas as pd
import numpy as np

# === 1. 讀取資料 ===
input_csv = 'scores.csv'
df = pd.read_csv(input_csv)


# 直接從 scores.csv 取三個分數欄位
res_spike_anomaly_score = df['res_spike_anomaly_score']
distance_score = df['distance_score']
slope_score = df['slope_score']
id = df['id']

