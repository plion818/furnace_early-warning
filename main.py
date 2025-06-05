import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import io # Required for reading UploadFile content

# Import the detection function from anomaly_detection.py
try:
    from anomaly_detection import detect_resistance_outliers_by_window
except ImportError:
    # This fallback is for environments where the file might be in a different relative path or for testing
    # In a proper package structure, the direct import should work.
    try:
        from .anomaly_detection import detect_resistance_outliers_by_window # if anomaly_detection is in the same directory
    except ImportError:
        raise ImportError("Could not import detect_resistance_outliers_by_window from anomaly_detection.py")


app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting anomalies in sensor data using a sliding window approach.",
    version="1.0.0"
)

@app.post("/detect_anomaly/")
async def create_detection_task(
    file: UploadFile = File(..., description="CSV file containing 'resistance' and 'resistance_scaled' columns."),
    window_size: int = Query(50, gt=0, description="The number of data points in the sliding window."),
    z_thresh: float = Query(3.0, gt=0, description="The Z-score threshold to identify outliers."),
    stride: int = Query(1, gt=0, description="The step size for moving the sliding window."),
    vote_threshold: float = Query(0.5, ge=0, le=1, description="Threshold (0 to 1) for the anomaly score to be considered a final anomaly.")
):
    """
    Detects anomalies in uploaded CSV data.

    The endpoint receives a CSV file and several parameters to control the anomaly detection process.
    - **file**: The CSV file to be processed. Must contain 'resistance' and 'resistance_scaled' columns.
    - **window_size**: Defines the size of the sliding window used for local outlier detection. Must be greater than 0.
    - **z_thresh**: The Z-score threshold. Data points with a Z-score (calculated within a window)
                  above this value are considered potential outliers. Must be greater than 0.
    - **stride**: The number of data points the window slides forward in each step. Must be greater than 0.
    - **vote_threshold**: A value between 0 and 1. A data point is confirmed as an anomaly if the ratio of
                        times it was flagged as an outlier (across all windows it appeared in)
                        to the total number of windows it appeared in, meets or exceeds this threshold.

    The API reads the CSV, performs validation, and then uses the `detect_resistance_outliers_by_window`
    function to identify anomalies.
    Returns a JSON response with the original filename, parameters used, a list of anomaly indices,
    and a list of anomaly scores for all data points.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Read file contents asynchronously
        contents = await file.read()
        # Create a pandas DataFrame from the byte contents
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        # Handle errors during file reading or CSV parsing
        raise HTTPException(status_code=400, detail=f"Error reading or parsing CSV file: {e}")

    # Validate required columns
    if not all(col in df.columns for col in ['resistance', 'resistance_scaled']):
        raise HTTPException(status_code=400, detail="CSV file must contain 'resistance' and 'resistance_scaled' columns.")

    # Handle empty DataFrame case
    if df.empty:
        return JSONResponse(content={
            "message": "Uploaded CSV file is empty.",
            "filename": file.filename,
            "parameters": {
                "window_size": window_size,
                "z_thresh": z_thresh,
                "stride": stride,
                "vote_threshold": vote_threshold
            },
            "anomaly_indices": [],
            "anomaly_scores": []
        })

    try:
        # Call the core anomaly detection function
        anomaly_indices, anomaly_scores = detect_resistance_outliers_by_window(
            df,
            window_size=window_size,
            z_thresh=z_thresh,
            stride=stride,
            vote_threshold=vote_threshold
        )

        # Prepare and return the successful response
        return {
            "filename": file.filename,
            "parameters": {
                "window_size": window_size,
                "z_thresh": z_thresh,
                "stride": stride,
                "vote_threshold": vote_threshold
            },
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": anomaly_scores.tolist() # Convert numpy array to list for JSON serialization
        }
    except ValueError as ve:
        # Handle specific ValueErrors raised by the detection function (e.g., invalid parameters)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch-all for other unexpected errors during the detection process
        # Log this error on the server for debugging
        # logger.error(f"Unexpected error during anomaly detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during anomaly detection: {e}")

# To run this app:
# 1. Save it as main.py
# 2. Ensure anomaly_detection.py is in the same directory.
# 3. Install FastAPI and Uvicorn: pip install fastapi uvicorn pandas
# 4. Run with Uvicorn: uvicorn main:app --reload
