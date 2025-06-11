import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import io

try:
    from anomaly_detection import detect_resistance_outliers_by_window
except ImportError:
    try:
        from .anomaly_detection import detect_resistance_outliers_by_window
    except ImportError:
        raise ImportError("Could not import detect_resistance_outliers_by_window from anomaly_detection.py")

app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting anomalies in sensor data using a sliding window approach. Requires 'resistance', 'resistance_scaled', and 'record Time' columns in CSV.",
    version="1.1.0" # Version bump
)

@app.post("/detect_anomaly/")
async def create_detection_task(
    file: UploadFile = File(..., description="CSV file containing 'resistance', 'resistance_scaled', and 'record Time' columns."),
    window_size: int = Query(..., gt=0, description="The number of data points in the sliding window. (必填)"),
    z_thresh: float = Query(..., gt=0, description="The Z-score threshold to identify outliers. (必填)"),
    stride: int = Query(..., gt=0, description="The step size for moving the sliding window. (必填)"),
    vote_threshold: float = Query(..., ge=0, le=1, description="Threshold (0 to 1) for the anomaly score. (必填)")
):
    """
    Detects anomalies in uploaded CSV data.

    The CSV file must contain 'resistance', 'resistance_scaled', and 'record Time' columns.
    The 'record Time' column is crucial for mapping anomalies back to specific time points.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        contents = await file.read()
        # Attempt to parse 'record Time' as datetime, but don't fail if it can't be parsed by default.
        # The anomaly detection module itself might handle various types for 'record Time'.
        # Pandas will attempt to infer types; 'record Time' often comes as string/object.
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or parsing CSV file: {e}")

    # Validate that all required columns are present
    required_columns = ['resistance', 'resistance_scaled', 'record Time']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise HTTPException(status_code=400, detail=f"CSV file must contain the following columns: {required_columns}. Missing: {missing_cols}")

    # Handle empty DataFrame case
    if df.empty:
        return JSONResponse(content={
            "filename": file.filename,
            "parameters": {
                "window_size": window_size,
                "z_thresh": z_thresh,
                "stride": stride,
                "vote_threshold": vote_threshold
            },
            "anomalies": [] # Return empty list for anomalies, consistent with new structure
        })

    try:
        # Call the detection function.
        # It now returns a list of dictionaries, each representing an anomaly.
        detected_anomalies_list = detect_resistance_outliers_by_window(
            df, # Pass the full DataFrame which includes 'record Time'
            window_size=window_size,
            z_thresh=z_thresh,
            stride=stride,
            vote_threshold=vote_threshold
        )

        # Structure the response
        return {
            "filename": file.filename,
            "parameters": {
                "window_size": window_size,
                "z_thresh": z_thresh,
                "stride": stride,
                "vote_threshold": vote_threshold
            },
            "anomalies": detected_anomalies_list # Directly use the list of anomaly dictionaries
        }
    except ValueError as ve:
        # Catch ValueErrors from anomaly_detection (e.g., missing columns, bad parameters)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch-all for other unexpected errors during detection
        # It's good practice to log such errors on the server side for debugging.
        # logger.error(f"An unexpected error occurred during anomaly detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during anomaly detection: {str(e)}")

# To run this app:
# 1. Save it as main.py
# 2. Ensure anomaly_detection.py is in the same directory.
# 3. Install FastAPI and Uvicorn: pip install fastapi uvicorn pandas python-multipart
# 4. Run with Uvicorn: uvicorn main:app --reload
