import cv2
import numpy as np

# Load the pre-trained face detection and helmet detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
helmet_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_helmet.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the colors for the lights
green = (0, 255, 0)
red = (0, 0, 255)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect helmets in the frame
    helmets = helmet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and helmets
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in helmets:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Check if a helmet is detected and turn the light on accordingly
    if len(helmets) > 0:
        cv2.circle(frame, (50, 50), 20, green, -1)
        cv2.circle(frame, (100, 50), 20, (0, 0, 0), -1)
    else:
        cv2.circle(frame, (50, 50), 20, (0, 0, 0), -1)
        cv2.circle(frame, (100, 50), 20, red, -1)

    # Display the resulting frame
    cv2.imshow('Face and Helmet Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
