import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_gaze_tracking_data(num_lines, cwt_spike_freq, start_datetime):
    np.random.seed(42)
    
    # Convert input datetime string to datetime object
    base_time = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    timestamps = [base_time + timedelta(milliseconds=i*33) for i in range(num_lines)]  # ~30 FPS

    # Simulate gaze tracking data
    frame_numbers = np.arange(num_lines)
    left_eye_x = np.random.randint(500, 800, num_lines)
    left_eye_y = np.random.randint(100, 400, num_lines)
    right_eye_x = left_eye_x + np.random.randint(-10, 10, num_lines)  # Slight variation
    right_eye_y = left_eye_y + np.random.randint(-10, 10, num_lines)

    # Pupil size fluctuations
    left_pupil_size = np.random.uniform(2.5, 5.0, num_lines)
    right_pupil_size = left_pupil_size + np.random.uniform(-0.5, 0.5, num_lines)

    # Blink detection (sporadic blinks)
    left_blink = np.random.choice([0, 1], num_lines, p=[0.95, 0.05])
    right_blink = left_blink.copy()

    # Gaze coordinates
    gaze_x = np.random.randint(400, 1200, num_lines)
    gaze_y = np.random.randint(100, 800, num_lines)

    # Head pose angles
    head_pose_x = np.random.uniform(-10, 10, num_lines)
    head_pose_y = np.random.uniform(-5, 5, num_lines)
    head_pose_z = np.random.uniform(-3, 3, num_lines)

    # Introduce Controlled Wavelet Transform (CWT) Spikes at user-defined frequency
    spike_indices = np.linspace(0, num_lines-1, num=cwt_spike_freq, dtype=int)
    gaze_x[spike_indices] += np.random.randint(-200, 200, cwt_spike_freq)
    gaze_y[spike_indices] += np.random.randint(-150, 150, cwt_spike_freq)

    # Compile into a DataFrame
    gaze_data = pd.DataFrame({
        'timestamp': timestamps,
        'frame_number': frame_numbers,
        'left_eye_x': left_eye_x,
        'left_eye_y': left_eye_y,
        'right_eye_x': right_eye_x,
        'right_eye_y': right_eye_y,
        'left_pupil_size': np.round(left_pupil_size, 2),
        'right_pupil_size': np.round(right_pupil_size, 2),
        'left_blink': left_blink,
        'right_blink': right_blink,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'head_pose_x': np.round(head_pose_x, 2),
        'head_pose_y': np.round(head_pose_y, 2),
        'head_pose_z': np.round(head_pose_z, 2),
        'marker': np.nan  # Placeholder for event markers
    })

    return gaze_data

def main():
    # User inputs
    num_lines = int(input("Enter the number of lines for the dataset: "))
    cwt_spike_freq = int(input("Enter the frequency of CWT spikes: "))
    start_datetime = input("Enter the start date and time (YYYY-MM-DD HH:MM:SS): ")

    # Generate dataset
    gaze_data = generate_gaze_tracking_data(num_lines, cwt_spike_freq, start_datetime)

    # Save to CSV
    filename = f"gaze_tracking_data_{start_datetime.replace(' ', '_').replace(':', '')}.csv"
    gaze_data.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")

if __name__ == "__main__":
    main()
