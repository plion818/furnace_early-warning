import pandas as pd
import numpy as np

def detect_resistance_outliers_by_window(df, window_size, z_thresh, stride, vote_threshold):
    """
    Detects resistance outliers in a DataFrame using a sliding window approach.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'resistance', 'resistance_scaled', and 'record Time' columns.
        window_size (int): The number of data points in the sliding window.
        z_thresh (float): The Z-score threshold to identify outliers.
        stride (int): The step size for moving the sliding window.
        vote_threshold (float): A threshold (0 to 1) for the anomaly score.

    Returns:
        list: A list of dictionaries, where each dictionary represents an anomaly and has keys:
              'index' (int): The original index of the anomaly in the DataFrame.
              'record_Time' (object): The value from the 'record Time' column for the anomaly.
              'score' (float): The calculated anomaly score for that point.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    # Updated column check to include 'record Time'
    if not all(col in df.columns for col in ['resistance', 'resistance_scaled', 'record Time']):
        raise ValueError("Input DataFrame must contain 'resistance', 'resistance_scaled', and 'record Time' columns.")
    # Allow np.number for numeric type checks to be more robust with pandas dtypes
    if not all(isinstance(arg, (int, float)) or isinstance(arg, np.number) for arg in [window_size, z_thresh, stride, vote_threshold]):
        raise TypeError("Parameters 'window_size', 'z_thresh', 'stride', and 'vote_threshold' must be numeric.")
    if window_size <= 0 or stride <= 0:
        raise ValueError("'window_size' and 'stride' must be positive.")
    if not (0 <= vote_threshold <= 1):
        raise ValueError("'vote_threshold' must be between 0 and 1.")

    resistance = df['resistance'].values
    resistance_scaled = df['resistance_scaled'].values
    n_points = len(df)
    anomaly_counts = np.zeros(n_points, dtype=int)
    window_hits = np.zeros(n_points, dtype=int)

    if n_points == 0:
        return [] # Return empty list as per new return type

    # Sliding window calculation
    for i in range(window_size, n_points + stride, stride):
        current_window_end = min(i, n_points)
        current_window_start = max(0, current_window_end - window_size)

        if current_window_end <= current_window_start: # Should not happen with correct range
            continue

        # Segment for current window
        window_res_segment = resistance[current_window_start:current_window_end]
        window_scaled_segment = resistance_scaled[current_window_start:current_window_end]

        # Step 1: Filter outliers based on 'resistance_scaled' for robust mean/std calculation
        mask = np.abs(window_scaled_segment) < 1 # Example: keep values where scaled variation is small
        filtered_res_in_window = window_res_segment[mask]

        # If less than 50% of data remains after filtering, or no data, skip this window
        if len(filtered_res_in_window) < (current_window_end - current_window_start) * 0.5 or len(filtered_res_in_window) == 0:
            continue

        # Step 2: Calculate mean and std dev from the filtered resistance values
        mean_res = np.mean(filtered_res_in_window)
        std_res = np.std(filtered_res_in_window, ddof=1) # ddof=1 for sample standard deviation

        if std_res == 0: # Avoid division by zero if all filtered values are the same
            continue

        # Step 3: Check original window data points against the calculated threshold
        for k in range(current_window_start, current_window_end):
            window_hits[k] += 1 # Mark that this point was part of a processed window
            if abs(resistance[k] - mean_res) > z_thresh * std_res:
                anomaly_counts[k] += 1 # Increment anomaly vote for this point

    # Calculate anomaly score for each data point
    anomaly_score_array = np.zeros(n_points, dtype=float) # Use float for scores
    for j in range(n_points):
        if window_hits[j] > 0:
            anomaly_score_array[j] = anomaly_counts[j] / window_hits[j]

    # Construct the list of anomaly dictionaries
    final_anomalies_output = []
    # Iterate through all points to check if they meet the anomaly criteria
    for idx in range(n_points):
        # A point is an anomaly if it was hit by windows and its score meets the threshold
        if window_hits[idx] > 0 and anomaly_score_array[idx] >= vote_threshold:
            final_anomalies_output.append({
                'index': int(idx),  # Original index in the DataFrame
                'record_Time': df.loc[idx, 'record Time'], # Get 'record Time' using original DataFrame index
                'score': float(anomaly_score_array[idx]) # The calculated anomaly score
            })

    return final_anomalies_output
