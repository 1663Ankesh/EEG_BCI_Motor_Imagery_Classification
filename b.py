import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Load the .mat file
mat = sio.loadmat('A01T.mat')

# Explore the structure of 'data' (you may need to adjust according to the actual structure)
eeg_data = mat['data'][0, 0]['X']  # EEG signals (trials, channels, samples)
labels = mat['data'][0, 0]['y']    # Labels (class of each trial)

# Preprocess EEG data
# Assuming data is in (trials, channels, samples) format
eeg_data = np.expand_dims(eeg_data, axis=-1)  # Add a dimension for channels if necessary

# Encode labels to categorical
labels = labels.ravel()  # Flatten the label array if necessary
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

# Define Multi-Scale CNN Model
def build_multiscale_cnn(input_shape):
    input_layer = layers.Input(shape=input_shape)

    # Branch 1: Small scale
    branch1 = layers.Conv2D(16, (1, 3), activation='relu', padding='same')(input_layer)
    branch1 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.MaxPooling2D((1, 2))(branch1)

    # Branch 2: Medium scale
    branch2 = layers.Conv2D(16, (1, 5), activation='relu', padding='same')(input_layer)
    branch2 = layers.Conv2D(32, (1, 5), activation='relu', padding='same')(branch2)
    branch2 = layers.MaxPooling2D((1, 2))(branch2)

    # Branch 3: Large scale
    branch3 = layers.Conv2D(16, (1, 7), activation='relu', padding='same')(input_layer)
    branch3 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(branch3)
    branch3 = layers.MaxPooling2D((1, 2))(branch3)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2, branch3], axis=-1)
    merged = layers.Flatten()(merged)
    
    # Fully connected layers
    fc = layers.Dense(64, activation='relu')(merged)
    output = layers.Dense(labels.shape[1], activation='softmax')(fc)  # Adjust output shape based on the number of classes

    model = models.Model(inputs=input_layer, outputs=output)
    return model

# Model configuration
input_shape = X_train.shape[1:]  # Shape of the input data (channels, samples, 1)
model = build_multiscale_cnn(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
