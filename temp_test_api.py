import requests
import os

# Hardcoded values since we can't load .env easily in this environment
API_URL = "http://127.0.0.1:8000/detect_anomaly/"
API_TOKEN = None # Assuming no token for now based on .env content I saw
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
            print(f"Total anomalies detected: {len(api_result.get('anomalies', []))}")
        else:
            print("\nFailed to get a valid response from the API.")
