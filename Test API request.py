
import requests
import os
from dotenv import load_dotenv

# 載入 .env 設定
load_dotenv()
API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
LOCAL_CSV_PATH = os.path.join("data/processed/sensorID_28_standardized.csv")

def call_anomaly_detection_api(
    file_path, window_size=200, z_thresh=3.0, stride=1, vote_threshold=0.5
):
    params = {
        "window_size": window_size,
        "z_thresh": z_thresh,
        "stride": stride,
        "vote_threshold": vote_threshold,
    }
    file_name = os.path.basename(file_path)
    response = None
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, "text/csv")}
            headers = {}
            if API_TOKEN:
                headers["Authorization"] = f"Bearer {API_TOKEN}"
            print(f"Calling API at {API_URL} with parameters: {params} and file: {file_name}")
            response = requests.post(API_URL, params=params, files=files, headers=headers)
            response.raise_for_status()
            return response.json()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if response is not None:
            print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}. Is the server running at {API_URL}?")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred with the request: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    if response is not None:
        print(f"Response status code (if available): {response.status_code}")
    return None

if __name__ == "__main__":
    print("--- Starting Client Example: Calling Anomaly Detection API with Local CSV ---")
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"CRITICAL ERROR: The specified CSV file does not exist: {LOCAL_CSV_PATH}")
        print("Please ensure the file is present at the correct path relative to where this script is run.")
        print("If the 'data' directory is not in the same location as this script, adjust LOCAL_CSV_PATH.")
    else:
        print(f"Attempting to process file: {LOCAL_CSV_PATH}")
        api_result = call_anomaly_detection_api(
            LOCAL_CSV_PATH,
            window_size=200,
            z_thresh=3.0,
            stride=1,
            vote_threshold=0.5
        )
        if api_result:
            print("\n--- API Response ---")
            print(f"Filename Processed: {api_result.get('filename')}")
            print(f"Parameters Used: {api_result.get('parameters')}")
            anomalies = api_result.get('anomalies', [])
            total_anomalies = len(anomalies)
            print(f"Total anomalies detected: {total_anomalies}")
            if total_anomalies > 0:
                print("Sample anomalies:")
                for item in anomalies[:5]:
                    print(item)
            else:
                print("No anomalies detected with the given parameters.")
        else:
            print("\nFailed to get a valid response from the API.")
            print("Please check server logs and ensure the API is running correctly and the file path is accessible.")

    print("\n--- Client Example Finished ---")
    print(f"""
To run this example:
1. Make sure the FastAPI server ('main.py') is running.
   You can run it using: uvicorn main:app --reload
   Ensure 'anomaly_detection.py' is in the same directory as 'main.py'.
2. Install required Python packages: pip install requests pandas (pandas might be needed if you extend result processing)
3. Ensure the data file is available at the path: '{LOCAL_CSV_PATH}'
   (Relative to where you run 'python client_example.py')
4. Run this script in a separate terminal: python client_example.py
""")


# import pandas as pd

# # 1. 讀取缺失樣本
# missing_df = pd.read_csv("data/processed/sensorID_28_missing_samples.csv")
# missing_df['record Time'] = pd.to_datetime(missing_df['record Time'])
# res_missing_times = set(missing_df[missing_df['resistance'] == 1]['record Time'])

# # 2. 取得 API 回傳異常點（假設已存在 anomalies 變數，且每筆有 'record_Time' 欄位）
# # 例如：anomalies = api_result.get('anomalies', [])
# anomaly_times = set(pd.to_datetime([a['record_Time'] for a in anomalies]))

# # 3. 統計
# res_missing_and_anomaly = res_missing_times & anomaly_times
# res_missing_and_normal = res_missing_times - anomaly_times

# print(f"resistance 缺失且被判定為異常的數量: {len(res_missing_and_anomaly)}")
# print(f"resistance 缺失但未被判定為異常的數量: {len(res_missing_and_normal)}")