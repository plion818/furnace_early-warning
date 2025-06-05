
import pandas as pd
import matplotlib.pyplot as plt

# 讀取CSV檔案
df = pd.read_csv("data/processed/sensorID_28_standardized.csv")


# 檢查temperature_scaled的絕對值是否有大於3
has_outlier = (df['temperature_scaled'].abs() > 3)

# 統計大於2和大於1的個數
count_gt_2 = (df['temperature_scaled'].abs() > 2).sum()
count_gt_1 = (df['temperature_scaled'].abs() > 1).sum()

if has_outlier.any():
    print('temperature_scaled 欄位有絕對值大於 3 的資料')
else:
    print('temperature_scaled 欄位所有絕對值都小於等於 3')

print(f"temperature_scaled 欄位有絕對值大於 2 的個數: {count_gt_2}")
print(f"temperature_scaled 欄位有絕對值大於 1 的個數: {count_gt_1}")

# 計算四個標準化欄位的平均與標準差
cols = ['current_scaled', 'voltage_scaled', 'resistance_scaled', 'temperature_scaled']
for col in cols:
    mean = df[col].mean()
    std = df[col].std()
    # 為避免顯示 -0.0000，取絕對值後再格式化顯示
    # -0.0000 只是浮點數誤差，與 0.0000 無實質差異
    print(f"{col} 平均值: {abs(mean):.4f}, 標準差: {abs(std):.4f}")

# 計算四個標準化欄位的平均與標準差
"""
繪製 temperature_scaled 的鐘形圖（直方圖）
"""

# # 取得標準化前的 mean 和 std
# mean = df['temperature'].mean()
# std = df['temperature'].std()


# -------- temperature_scaled --------
mean_temp = df['temperature'].mean()
std_temp = df['temperature'].std()
plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(df['temperature_scaled'], bins=30, color='skyblue', edgecolor='black', density=True)
plt.title('Standardized Temperature Distribution')
plt.xlabel('temperature_scaled')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.5)
for z in [-3, -2, -1, 0, 1, 2, 3]:
    real_temp = mean_temp + z * std_temp
    plt.axvline(x=z, color='gray', linestyle='--', alpha=0.3)
    plt.text(z, plt.ylim()[1]*0.9, f'{real_temp:.1f}°C',
             color='dimgray', ha='center', fontsize=9, rotation=0,
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'),
             va='bottom')
plt.show()

# # -------- resistance_scaled --------
# mean_res = df['resistance'].mean()
# std_res = df['resistance'].std()
# plt.figure(figsize=(8, 5))
# n, bins, patches = plt.hist(df['resistance_scaled'], bins=30, color='lightgreen', edgecolor='black', density=True)
# plt.title('Standardized Resistance Distribution')
# plt.xlabel('resistance_scaled')
# plt.ylabel('Density')
# plt.grid(True, linestyle='--', alpha=0.5)
# for z in [-3, -2, -1, 0, 1, 2, 3]:
#     real_res = mean_res + z * std_res
#     plt.axvline(x=z, color='gray', linestyle='--', alpha=0.3)
#     plt.text(z, plt.ylim()[1]*0.9, f'{real_res:.1f}',
#              color='dimgray', ha='center', fontsize=9, rotation=0,
#              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'),
#              va='bottom')
# plt.show()