import pandas as pd

class AnomalyMarker:
    def __init__(self, missing_csv_path):
        self.missing_df = pd.read_csv(missing_csv_path)
        self.missing_df['record Time'] = pd.to_datetime(self.missing_df['record Time'])

    def mark_res_missing(self, df):
        """
        在 df 中新增 is_res_missing_anomaly 欄位，
        若該點在缺失csv中且 resistance=1，則為 True。
        """
        df = df.copy()
        # 只取 resistance 缺失的時間
        res_missing_times = set(self.missing_df[self.missing_df['resistance'] == 1]['record Time'])
        df['is_res_missing_anomaly'] = df['record Time'].isin(res_missing_times)
        return df
