import pandas as pd

# Load the CSV file
file_path = "data/raw/加熱爐資料(to 0410).csv"
df = pd.read_csv(file_path, header=None)

# Insert column names
df.columns = ['id', 'sensorID', 'current', 'voltage', 'resistance', 'temperature', 'record Time']

# Convert 'record Time' to datetime format
df['record Time'] = pd.to_datetime(df['record Time'], errors='coerce')


# Filter rows where sensorID == 28
filtered_df = df[df['sensorID'] == 28].copy()

# 檢查 record Time 是否為 datetime，若不是則轉換
if not pd.api.types.is_datetime64_any_dtype(filtered_df['record Time']):
    filtered_df['record Time'] = pd.to_datetime(filtered_df['record Time'], errors='coerce')

# 計算時間間隔
filtered_df = filtered_df.sort_values('record Time')
intervals = filtered_df['record Time'].diff().dropna()
print('所有間隔是否一致:', intervals.nunique() == 1)

# Save the filtered data to a new file
filtered_file_path = "data/processed/sensorID_28_filtered.csv"
filtered_df.to_csv(filtered_file_path, index=False)

print("Filtered data (first 5 rows):", filtered_df.head())
print("Total rows after filtering:", len(filtered_df))
print("Filtered data saved to:", filtered_file_path)




