import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -------------------------------
# Simulate time series
# -------------------------------
np.random.seed(42)
data_length = 100
data = np.random.normal(0, 1, data_length)

# Inject anomalies
anomalies_idx = [20, 45, 70]
data[anomalies_idx] += np.random.normal(8, 1, len(anomalies_idx))

# -------------------------------
# Thresholds
# -------------------------------
mean = np.mean(data[:10])
std = np.std(data[:10])
upper_thresh = mean + 3 * std
lower_thresh = mean - 3 * std

# Rolling window size
window_size = 30

# -------------------------------
# Setup figure
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, window_size)
ax.set_ylim(min(data) - 2, max(data) + 2)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("Real-Time Anomaly Detection in Time Series")

line, = ax.plot([], [], lw=2, color='blue', label='Data')
anomaly_points, = ax.plot([], [], 'ro', markersize=8, label='Anomaly')
text_status = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                      fontsize=14, weight='bold', ha='center', va='top')
ax.legend(loc='upper left')

# Global variables to track data and shaded area
x_data, y_data = [], []
anomaly_x, anomaly_y = [], []
shaded = None


# -------------------------------
# Update function
# -------------------------------
def update(frame):
    global shaded

    x_data.append(frame)
    y_data.append(data[frame])

    # Rolling window
    start = max(0, frame - window_size)
    ax.set_xlim(start, start + window_size)

    # Detect anomalies
    if data[frame] > upper_thresh or data[frame] < lower_thresh:
        anomaly_x.append(frame)
        anomaly_y.append(data[frame])
        text_status.set_text(f'Anomaly Detected! Value={data[frame]:.2f}')
        text_status.set_color('red')  # Red for anomaly
    else:
        text_status.set_text(f'Normal Value={data[frame]:.2f}')
        text_status.set_color('green')  # Red for anomaly

    # Update line and anomalies
    line.set_data(x_data[start:frame + 1], y_data[start:frame + 1])
    anomaly_points.set_data(anomaly_x, anomaly_y)

    # Remove previous shaded area
    if shaded:
        shaded.remove()
    shaded = ax.fill_between(range(start, frame + 1), lower_thresh, upper_thresh, color='green', alpha=0.1)

    return line, anomaly_points, text_status, shaded


ani = FuncAnimation(fig, update, frames=data_length, blit=True, repeat=False)
ani.save("real_time_anomaly.gif", writer=PillowWriter(fps=10))

