import cv2
from fer import FER
import numpy as np

# Initialize FER emotion detector
emotion_detector = FER()

# Initialize OpenCV's pre-trained Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each detected face, detect emotions
    for (x, y, w, h) in faces:
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        # Detect emotion in the face region
        emotion, score = emotion_detector.top_emotion(face_region)
        
        # Only track 'anxiety' or 'sadness'
        if emotion in ['fear', 'sadness','happy','shocked']:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
            cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected faces and emotions
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
