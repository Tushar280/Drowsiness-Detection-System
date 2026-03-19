import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("drowsiness_detection_model.keras")

print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("drowsiness_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete! Saved as drowsiness_model.tflite")
