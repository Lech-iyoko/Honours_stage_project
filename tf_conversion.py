import tensorflow as tf

# Load the saved TensorFlow model
model = tf.keras.models.load_model('~/~catkin_ws2/src/face_recognition/face_recognition_model')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model_file('~/~catkin_ws2/src/face_recognition/face_recognition_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)

