Eye Tracking Mouse using OpenCV – Project Explanation & Execution
This project allows you to control the mouse cursor using eye movements. We use OpenCV, dlib (for face & eye detection), and PyAutoGUI (for mouse control).

1. Project Overview Detects face and eyes using OpenCV & dlib. Tracks eye movement direction (left, right, up, down). Moves the mouse pointer based on eye direction.

2. Requirements Before starting, install the required libraries:bash Copy code pip install opencv-python dlib numpy pyautogui imutils

3. Step-by-Step Execution

Step 1: Import Libraries python Copy code
import cv2
import dlib
import numpy as np
import pyautogui

Step 2: Load Face & Eye Detection Model python Copy code
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib website

Step 3: Define Eye Tracking Function python Copy code
def get_eye_region(landmarks, eye_points):
    eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points])
    return eye_region

Step 4: Process Webcam Feed & Track Eyes python Copy code
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])

        # Find eye center
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)

        # Move mouse based on eye movement
        pyautogui.moveTo(left_eye_center[0] * 4, left_eye_center[1] * 3)

        # Draw eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

    cv2.imshow("Eye Tracking Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

5. How to Run the Project?
Download dlib's shape predictor model
Download shape_predictor_68_face_landmarks.dat
Extract and place in your project folder.
Run the Python script bash Copy code
python eye_tracking_mouse.py
Control the mouse with your eyes!
Move your eyes left/right → Mouse moves accordingly.
Blink or hold gaze → Click functionality (can be added).

6. Future Improvements
Improve accuracy using deep learning (CNN-based eye tracking).
Add clicking actions using blinking detection.
Optimize performance for real-time tracking.
