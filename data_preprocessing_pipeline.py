
import pandas as pd
import numpy as np

# Load captured raw data
raw_data = pd.read_csv('captured_data.csv')

# Label assignment logic (assuming manual labeling or separate labeled attacks)
# For illustration purposes, let's randomly assign labels (to be replaced with actual labeling logic)
raw_data['label'] = np.random.choice([0, 1], size=len(raw_data), p=[0.9, 0.1])  # 0: Legitimate, 1: Malicious

# Feature engineering: packet frequency per MAC, packet types/subtypes frequency, inter-arrival times
features = []

# Calculate frequency-based features
mac_counts = raw_data['Src_MAC'].value_counts().to_dict()
raw_data['MAC_freq'] = raw_data['Src_MAC'].map(mac_counts)

# Inter-arrival time calculation
raw_data['inter_arrival'] = raw_data['Time'].diff().fillna(0)

# Aggregated statistical features
aggregated_features = raw_data.groupby('Src_MAC').agg({
    'MAC_freq': 'mean',
    'inter_arrival': ['mean', 'std'],
    'Signal': ['mean', 'std'],
}).reset_index()

aggregated_features.columns = ['Src_MAC', 'MAC_freq_mean', 'inter_arrival_mean', 'inter_arrival_std', 'signal_mean', 'signal_std']

# Merge aggregated features back to original data
preprocessed_data = pd.merge(raw_data, aggregated_features, on='Src_MAC', how='left')

# Drop irrelevant or raw columns not needed for ML
final_features = preprocessed_data.drop(['Time', 'Src_MAC', 'Dst_MAC', 'Signal'], axis=1)

# Save preprocessed and labeled dataset
final_features.to_csv('../datasets/preprocessed_features.csv', index=False)
print("Dataset preprocessing and feature extraction completed successfully.")
