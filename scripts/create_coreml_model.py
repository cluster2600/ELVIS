import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import coremltools as ct
import os

# Define input features (matching your REQUIRED_FEATURES)
NUM_FEATURES = 20  # Length of REQUIRED_FEATURES
CLASSES = ["BUY", "HOLD", "SELL"]

# Generate dummy data for demonstration
def generate_dummy_data(samples=1000):
    X = np.random.random((samples, NUM_FEATURES)).astype(np.float32)
    y = np.random.randint(0, 3, size=(samples,))  # 0: BUY, 1: HOLD, 2: SELL
    return X, tf.keras.utils.to_categorical(y, num_classes=3)

# Create a simple neural network with explicit input name
def create_model():
    inputs = layers.Input(shape=(NUM_FEATURES,), name="input")  # Explicitly name the input
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(3, activation='softmax', name="output")(x)  # 3 classes
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save the model
X_train, y_train = generate_dummy_data()
model = create_model()
model.summary()  # Print model summary to verify input name
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Convert to Core ML
coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(name="input", shape=(1, NUM_FEATURES))],  # Match the input name
    classifier_config=ct.ClassifierConfig(class_labels=CLASSES),
    minimum_deployment_target=ct.target.iOS14  # Ensure compatibility with older systems
)

# Save the Core ML model
output_path = "/Users/maxime/BTC_BOT/models/trading_model.mlmodel"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
coreml_model.save(output_path)
print(f"Core ML model saved to {output_path}")