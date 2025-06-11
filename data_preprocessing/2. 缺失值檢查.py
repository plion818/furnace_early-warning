import pandas as pd

# 讀取 sensorID_28_filtered.csv 檔案
file_path = "data/processed/sensorID_28_filtered.csv"
df = pd.read_csv(file_path)

columns = df.columns.tolist()

# 將 '0' 與 'n/a' 視為缺失值
na_values = ['0', 'n/a', 'N/A', 0]
df = df.replace(na_values, pd.NA)

# 計算每一欄的缺失值個數
missing_counts = df.isna().sum()
print('每一欄的缺失值個數:')
print(missing_counts)

# 顯示每一欄前幾個缺失值的資料列
print('\n每一欄前幾個缺失值的資料列:')
for col in columns:
    missing_rows = df[df[col].isna()]
    if not missing_rows.empty:
        print(f"\n欄位: {col}")
        print(missing_rows.head(3))

# 缺失值填補：用前一筆有數值的資料填補 (forward fill)
df_filled = df.ffill()
df_filled = df_filled.infer_objects(copy=False)


# 再次計算每一欄的缺失值個數
print('\n填補後每一欄的缺失值個數:')
print(df_filled.isna().sum())

# 將填補後的資料導出為新的 CSV 檔案
# output_path = "data/processed/sensorID_28_missing_filled.csv"
# df_filled.to_csv(output_path, index=False)
# print(f"\n填補後的資料已儲存至: {output_path}")

# 額外匯出缺失樣本狀態
missing_cols = ['current', 'voltage', 'resistance', 'temperature']
output_missing = []
for idx, row in df.iterrows():
    record = {'record Time': row.get('record Time', None)}
    has_missing = False
    for col in missing_cols:
        is_missing = int(pd.isna(row.get(col, None)))
        record[col] = is_missing
        if is_missing:
            has_missing = True
    if has_missing:
        output_missing.append(record)

if output_missing:
    missing_df = pd.DataFrame(output_missing)
    missing_output_path = "data/processed/sensorID_28_missing_samples.csv"
    missing_df.to_csv(missing_output_path, index=False)
    print(f"\n缺失樣本狀態已儲存至: {missing_output_path}")
