import requests
import pandas as pd
import io

# Define the API endpoint URL
API_URL = "http://127.0.0.1:8000/detect_anomaly/"

def create_sample_csv():
    """Creates an in-memory sample CSV file for testing."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00',
                                     '2023-01-01 00:03:00', '2023-01-01 00:04:00', '2023-01-01 00:05:00',
                                     '2023-01-01 00:06:00', '2023-01-01 00:07:00', '2023-01-01 00:08:00',
                                     '2023-01-01 00:09:00']),
        'resistance': [10, 10.2, 10.1, 9.9, 15.0, 10.3, 10.0, 9.8, 10.2, 10.1], # Anomaly at index 4
        'resistance_scaled': [0.1, 0.3, 0.2, 0.0, 5.0, 0.4, 0.1, -0.1, 0.3, 0.2] # Scaled anomaly
    }
    df = pd.DataFrame(data)

    # Create an in-memory CSV file using io.StringIO
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0) # Rewind the buffer to the beginning to be read later
    return "sample_data.csv", csv_buffer

def call_anomaly_detection_api(file_name, file_object, window_size=5, z_thresh=2.0, stride=1, vote_threshold=0.5):
    """
    Calls the anomaly detection API with a file and parameters.

    Args:
        file_name (str): The name of the file to be sent.
        file_object (io.StringIO or io.BytesIO): The file object containing CSV data.
                                                 The file object should be seeked to the beginning if it was read before.
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
        "vote_threshold": vote_threshold
    }

    # Prepare the file for the multipart/form-data request
    # 'file' is the name of the field expected by the FastAPI endpoint for the UploadFile
    files = {
        "file": (file_name, file_object, "text/csv") # (filename, file-like object, content_type)
    }

    print(f"Calling API at {API_URL} with parameters: {params} and file: {file_name}")

    response = None # Initialize response to None for broader scope in error handling
    try:
        # Make the POST request
        response = requests.post(API_URL, params=params, files=files)
        # Raise an HTTPError for bad responses (4XX or 5XX client/server errors)
        response.raise_for_status()
        return response.json()  # Return the parsed JSON response
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

    # If any exception occurred, print details if available
    if response is not None:
        print(f"Response status code: {response.status_code}")
    return None # Return None if any error occurred

if __name__ == "__main__":
    print("--- Starting Client Example ---")

    # 1. Create sample CSV data in-memory
    sample_file_name, sample_csv_buffer = create_sample_csv()
    print(f"Generated sample CSV: {sample_file_name}")

    # 2. Define detection parameters.
    # These are chosen to likely detect the anomaly (15.0) in the sample data.
    # The value 15.0 is at index 4.
    # With window_size=5, a window like [10.1, 9.9, 15.0, 10.3, 10.0] will be analyzed.
    # The 'resistance_scaled' for 15.0 is 5.0, which will be used for initial filtering.
    # If 15.0 passes initial filtering (abs(5.0) < 1 is false), it means the logic in
    # detect_resistance_outliers_by_window related to 'resistance_scaled' filtering needs checking.
    # Assuming the provided function's logic:
    # The `mask = np.abs(window_scaled) < 1` would filter out 15.0 if `resistance_scaled` is high.
    # For the purpose of this client, we assume `resistance_scaled` is such that the point of interest (15.0) *is not* initially filtered out by the scaled value check,
    # or that the window contains enough other points to proceed.
    # Let's assume the primary detection of 15.0 being an outlier is based on its 'resistance' value
    # compared to the mean of *filtered* values in the window.

    custom_window_size = 5
    custom_z_thresh = 1.5      # Lower z_thresh to be more sensitive for the small, clear anomaly.
    custom_stride = 1
    custom_vote_threshold = 0.1 # Lower vote_threshold; the anomaly point will be in fewer windows.

    # 3. Call the API
    print("\n--- Calling Anomaly Detection API ---")
    api_result = call_anomaly_detection_api(
        sample_file_name,
        sample_csv_buffer, # Pass the buffer object
        window_size=custom_window_size,
        z_thresh=custom_z_thresh,
        stride=custom_stride,
        vote_threshold=custom_vote_threshold
    )

    # 4. Print the results
    if api_result:
        print("\n--- API Response ---")
        print(f"Filename: {api_result.get('filename')}")
        print(f"Parameters Used: {api_result.get('parameters')}")

        anomaly_indices = api_result.get('anomaly_indices', [])
        anomaly_scores = api_result.get('anomaly_scores', []) # This is a list

        print(f"Anomaly Indices Found: {anomaly_indices}")
        # print(f"Full Anomaly Scores: {anomaly_scores}") # Can be very long, uncomment if needed

        if anomaly_indices:
            print("\nDetails of detected anomalies from original data:")
            # Reset buffer to read the original data for displaying values
            sample_csv_buffer.seek(0)
            df_original = pd.read_csv(sample_csv_buffer)

            for index in anomaly_indices:
                if 0 <= index < len(df_original) and index < len(anomaly_scores):
                    print(f"  - Index {index}: resistance = {df_original.iloc[index]['resistance']}, "
                          f"original_scaled = {df_original.iloc[index]['resistance_scaled']}, "
                          f"calculated_score = {anomaly_scores[index]:.4f}")
                else:
                    print(f"  - Index {index}: (out of bounds for original data or scores array)")
        else:
            print("No anomalies detected with the given parameters and data.")
            print("Consider adjusting parameters or checking the data if anomalies were expected.")
    else:
        print("\nFailed to get a valid response from the API.")

    # Clean up: Close the in-memory file buffer
    if hasattr(sample_csv_buffer, 'close'):
        sample_csv_buffer.close()
    print("\nSample CSV buffer closed.")

    print("\n--- Client Example Finished ---")
    print(f"""
To run this example:
1. Ensure the FastAPI server ('main.py') is running.
   You can run it using: uvicorn main:app --reload
   Make sure 'anomaly_detection.py' is in the same directory as 'main.py'.
2. Install required Python packages: pip install requests pandas
3. Run this script in a separate terminal: python client_example.py
""")
