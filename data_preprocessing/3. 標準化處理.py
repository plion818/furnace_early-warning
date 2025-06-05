import pandas as pd
from sklearn.preprocessing import StandardScaler

# 讀取填補缺失值後的資料
file_path = "data/processed/sensorID_28_missing_filled.csv"
df = pd.read_csv(file_path)

# 需要標準化的欄位
cols_to_scale = ['current', 'voltage', 'resistance', 'temperature']


# 建立標準化物件並進行標準化，並保留原始欄位
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[cols_to_scale])
df_with_scaled = df.copy()
for i, col in enumerate(cols_to_scale):
    df_with_scaled[f'{col}_scaled'] = scaled_values[:, i]

# 將同時包含原始與標準化欄位的資料存成新檔案
output_path = "data/processed/sensorID_28_standardized.csv"
df_with_scaled.to_csv(output_path, index=False)
print(f"標準化後的資料已儲存至: {output_path}")
