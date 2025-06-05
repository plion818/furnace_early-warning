import pandas as pd
import numpy as np

def detect_resistance_outliers_by_window(df, window_size, z_thresh, stride, vote_threshold):
    """
    Detects resistance outliers in a DataFrame using a sliding window approach.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'resistance' and 'resistance_scaled' columns.
        window_size (int): The number of data points in the sliding window.
        z_thresh (float): The Z-score threshold to identify outliers.
                           A point is considered an outlier if its deviation from the window's mean
                           (after initial filtering) is greater than z_thresh * standard_deviation.
        stride (int): The step size for moving the sliding window.
        vote_threshold (float): A threshold (0 to 1) for the anomaly score.
                                If a data point's anomaly score (ratio of times it's flagged as an
                                outlier to the number of windows it appeared in) is greater than or
                                equal to this threshold, it's considered a final anomaly.

    Returns:
        tuple: A tuple containing:
            - final_anomalies (list): A list of indices of the data points identified as anomalies.
            - anomaly_score (np.ndarray): An array of anomaly scores for each data point.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['resistance', 'resistance_scaled']):
        raise ValueError("Input DataFrame must contain 'resistance' and 'resistance_scaled' columns.")
    if not all(isinstance(arg, (int, float)) for arg in [window_size, z_thresh, stride, vote_threshold]):
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
        return [], np.array([])

    for i in range(window_size, n_points + stride, stride): # Adjusted loop to correctly cover data points
        current_window_end = min(i, n_points)
        current_window_start = max(0, current_window_end - window_size)

        if current_window_end <= current_window_start: # Should not happen with correct logic but as a safe guard
            continue

        # Get data for the current window
        window_res = resistance[current_window_start:current_window_end]
        window_scaled = resistance_scaled[current_window_start:current_window_end]

        # Step 1: Filter outliers based on 'resistance_scaled'
        # Mask to keep values where absolute scaled resistance is less than 1
        mask = np.abs(window_scaled) < 1
        filtered_res_in_window = window_res[mask]

        # If less than 50% of data remains after filtering, skip this window
        if len(filtered_res_in_window) < (current_window_end - current_window_start) * 0.5:
            continue

        # Step 2: Calculate mean and std dev from the filtered resistance values
        mean_res = np.mean(filtered_res_in_window)
        std_res = np.std(filtered_res_in_window, ddof=1) # ddof=1 for sample standard deviation

        if std_res == 0: # Avoid division by zero if all values are the same
            continue

        # Step 3: Check original window data points against the calculated threshold
        for k in range(current_window_start, current_window_end):
            window_hits[k] += 1
            if abs(resistance[k] - mean_res) > z_thresh * std_res:
                anomaly_counts[k] += 1

    # Calculate anomaly score for each data point
    anomaly_score = np.zeros(n_points)
    for j in range(n_points):
        if window_hits[j] > 0:
            anomaly_score[j] = anomaly_counts[j] / window_hits[j]

    # Identify final anomalies based on the vote_threshold
    final_anomalies = [idx for idx, score in enumerate(anomaly_score) if window_hits[idx] > 0 and score >= vote_threshold]

    return final_anomalies, anomaly_score
