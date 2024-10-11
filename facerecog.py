import cv2
import numpy as np
from keras.models import model_from_json

# Load model architecture
with open('config.json', 'r') as f:
    config = f.read()
model = model_from_json(config)

# Load model weights
model.load_weights('model.weights.h5')

# Load the label names
target_names = ['Lech', 'Alex', 'Imaan', 'Nick']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the confidence threshold for predictions
confidence_threshold = 0.8

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face to match the model's input size
        face_resized = cv2.resize(face, (160, 160))

        # Preprocess the face image for the model
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0

        # Perform face recognition using the model
        embedding = model.predict(face_resized)
        predicted_class = np.argmax(embedding)
        confidence = np.max(embedding)

        # Check if confidence is above the threshold
        if confidence > confidence_threshold and predicted_class < len(target_names):
            predicted_name = target_names[predicted_class]
        else:
            predicted_name = 'Unknown'

        # Draw bounding box around the face and label it with confidence
        label_text = '{} ({:.2f})'.format(predicted_name, confidence)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

