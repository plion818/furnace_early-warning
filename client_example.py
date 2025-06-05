import requests
import os # For joining paths
import pandas as pd # Still used for displaying results if needed, though not for sample creation

# Define the API endpoint URL
API_URL = "http://127.0.0.1:8000/detect_anomaly/"
# Define the path to the local CSV file
# Assuming 'data/processed/' is relative to the script's location or a known base path.
# If the script is at the root of the repo, this path should work.
LOCAL_CSV_PATH = os.path.join("data", "processed", "sensorID_28_standardized.csv")


def call_anomaly_detection_api(
    file_path, window_size=50, z_thresh=3.0, stride=1, vote_threshold=0.5
):
    """
    Calls the anomaly detection API with a local file and parameters.

    Args:
        file_path (str): The path to the local CSV file.
        window_size (int): Detection parameter for the sliding window size.
        z_thresh (float): Detection parameter for the Z-score threshold.
        stride (int): Detection parameter for the window stride.
        vote_threshold (float): Detection parameter for the anomaly vote threshold.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    # Define the query parameters for the API request
    params = {
        "window_size": window_size,
        "z_thresh": z_thresh,
        "stride": stride,
        "vote_threshold": vote_threshold,
    }

    # Extract the filename from the path to be used in the multipart request
    file_name = os.path.basename(file_path)

    response = None # Initialize response to None for broader scope in error handling
    try:
        # Open the file in binary read mode ("rb")
        with open(file_path, "rb") as f:
            # Prepare the file for the multipart/form-data request
            files = {"file": (file_name, f, "text/csv")} # (filename, file-like object, content_type)

            print(f"Calling API at {API_URL} with parameters: {params} and file: {file_name}")
            # Make the POST request
            response = requests.post(API_URL, params=params, files=files)
            # Raise an HTTPError for bad responses (4XX or 5XX client/server errors)
            response.raise_for_status()
            return response.json()  # Return the parsed JSON response

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
        # This catches other request-related errors, e.g., issues with the URL
        print(f"An unexpected error occurred with the request: {req_err}")
    except Exception as e:
        # Catch any other unexpected errors (e.g., issues not directly from `requests`)
        print(f"An unexpected error occurred: {e}")

    # If any exception occurred and response object exists, print details
    if response is not None: # This check might be redundant if error handling above is exhaustive
        print(f"Response status code (if available): {response.status_code}")
    return None # Return None if any error occurred

if __name__ == "__main__":
    print("--- Starting Client Example: Calling Anomaly Detection API with Local CSV ---")

    # Check if the target CSV file exists before attempting to call the API
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"CRITICAL ERROR: The specified CSV file does not exist: {LOCAL_CSV_PATH}")
        print("Please ensure the file is present at the correct path relative to where this script is run.")
        print("If the 'data' directory is not in the same location as this script, adjust LOCAL_CSV_PATH.")
    else:
        print(f"Attempting to process file: {LOCAL_CSV_PATH}")
        # Call the API with the specified local file and parameters
        # Parameters for the actual dataset:
        api_result = call_anomaly_detection_api(
            LOCAL_CSV_PATH,
            window_size=50,   # Default from function signature
            z_thresh=3.0,     # Default from function signature
            stride=1,         # Default from function signature
            vote_threshold=0.5 # Default from function signature
        )

        # Print the results
        if api_result:
            print("\n--- API Response ---")
            print(f"Filename Processed: {api_result.get('filename')}")
            print(f"Parameters Used: {api_result.get('parameters')}")

            anomaly_indices = api_result.get('anomaly_indices', [])
            total_anomalies = len(anomaly_indices)

            print(f"Total anomalies detected: {total_anomalies}")

            if total_anomalies > 0:
                print(f"First 20 anomaly indices: {anomaly_indices[:20]}")
                # Optionally, print more details about anomalies if needed,
                # For example, loading the CSV with pandas and showing values at these indices.
                # This part is omitted for brevity here but was present in the sample data version.
            else:
                print("No anomalies detected with the given parameters.")

            # To see all anomaly scores (can be very long):
            # anomaly_scores = api_result.get('anomaly_scores', [])
            # print(f"First 20 anomaly scores: {anomaly_scores[:20]}")

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
