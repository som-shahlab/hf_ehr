import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
input_parquet_path = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/eda/df__starr__inter_event_times.parquet'

# Load the Parquet file
df = pd.read_parquet(input_parquet_path)
sns.set_style('whitegrid', rc={
    'xtick.bottom': True,
    'ytick.left': True,
})

# Group by 'pid' to calculate the mean and standard deviation of time between events for each patient
df_grouped = df.groupby('pid').agg({'time': ['mean', 'std', lambda x: np.percentile(x, 75) - np.percentile(x, 25)]}).reset_index()
df_grouped.columns = ['pid', 'mean_time_between_events', 'std_time_between_events', 'iqr_time_between_events']

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Bar height (maximum frequency) for visual alignment
bar_height_mean = 25000
bar_height_std = 40000
bar_height_iqr = 20000

# Define consistent width relative to log scale
relative_width = 0.02  # Adjust this to get the desired visual width

# Plot 1: Histogram of mean times between events across all patients
sns.histplot(df_grouped['mean_time_between_events'], bins=50, kde=True, log_scale=(True, False), ax=axes[0], color='blue', label='EHR Data')
axes[0].set_title('Mean Times Between Events')
axes[0].set_xlabel('Mean Time Between Events (seconds)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)
# Add NLP mean time overlay (a red bar at mean = 1)
axes[0].bar(x=1, height=bar_height_mean, width=1 * relative_width, color='red', label='NLP Mean Time = 1')

# Plot 2: Histogram of standard deviation of time between events within each patient
sns.histplot(df_grouped['std_time_between_events'], bins=50, kde=True, log_scale=(True, False), ax=axes[1], color='blue', label='EHR Data')
axes[1].set_title('Standard Deviation of Times Between Events')
axes[1].set_xlabel('Standard Deviation of Time Between Events (seconds)')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)
# Add NLP std deviation overlay (a red bar close to zero, at x = 1e-2)
axes[1].bar(x=1e-2, height=bar_height_std, width=1e-2 * relative_width, color='orange', label='NLP Std Dev ≈ 0')

# Plot 3: Histogram of IQR of time between events within each patient
sns.histplot(df_grouped['iqr_time_between_events'], bins=50, kde=True, log_scale=(True, False), ax=axes[2], color='blue', label='EHR Data')
axes[2].set_title('IQR of Times Between Events')
axes[2].set_xlabel('IQR of Time Between Events (seconds)')
axes[2].set_ylabel('Frequency')
axes[2].grid(True)
# Add NLP IQR overlay (a red bar close to zero, at x = 1e-2)
axes[2].bar(x=1e-2, height=bar_height_iqr, width=1e-2 * relative_width, color='red', label='NLP IQR ≈ 0')

# Add legends for each subplot
for ax in axes:
    ax.legend(loc='upper right')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'histogram_time_between_events_with_NLP_overlay_bars_fixed_width_zero.png')
plt.show()
