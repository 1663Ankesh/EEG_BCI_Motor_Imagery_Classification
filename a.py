import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the .mat file
mat = sio.loadmat('A01T.mat')

# Extract EEG data and labels
# Adjust the keys based on the structure of the .mat file
eeg_data = mat['data']  # Replace 'data' with the actual key if different
labels = mat['labels']  # Replace 'labels' with the actual key if different

# Preprocess data
# Assuming data is in (trials, channels, samples) format
eeg_data = np.expand_dims(eeg_data, axis=-1)  # Add a dimension for channels if necessary

# Encode labels to categorical
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

# Print shapes for verification
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")
print(f"Training Labels Shape: {y_train.shape}")
