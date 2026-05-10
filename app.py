import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import threading
import winsound
import math


CLASSES = ['Alert', 'Drowsy_Eyes_Closed', 'Drowsy_Yawning']
IMG_SIZE = 64
MODEL_PATH = "drowsiness_detection_model.keras"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model '{MODEL_PATH}' not found in the current directory.")
    print("Please run 'train.py' first.")
    exit()

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper function to crop bounding box safely from landmarks
def get_roi_from_landmarks(frame, landmarks, point_indices, padding=0.2):
    h_frame, w_frame = frame.shape[:2]
    pts = np.array([(int(landmarks.landmark[i].x * w_frame), int(landmarks.landmark[i].y * h_frame)) for i in point_indices])
    x, y, w, h = cv2.boundingRect(pts)
    
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    
    return frame[y1:y2, x1:x2]

# Helper function to calculate head tilt angle
def calculate_tilt(landmarks, frame):
    h, w = frame.shape[:2]
    # Left eye (image right): indices [362, 263, 386, 374]
    left_eye_pts = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in [362, 263, 386, 374]]
    # Right eye (image left): indices [33, 133, 159, 145]
    right_eye_pts = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in [33, 133, 159, 145]]
    
    left_cx = sum(pt[0] for pt in left_eye_pts) / 4.0
    left_cy = sum(pt[1] for pt in left_eye_pts) / 4.0
    
    right_cx = sum(pt[0] for pt in right_eye_pts) / 4.0
    right_cy = sum(pt[1] for pt in right_eye_pts) / 4.0
    
    dY = left_cy - right_cy
    dX = left_cx - right_cx
    
    angle = math.degrees(math.atan2(dY, dX))
    return angle

# Start video capture
cap = cv2.VideoCapture(0)

# Settings matching your required UI functionality
score = 0
THRESHOLD = 15
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1
thickness = 2
color_text = (255, 255, 255) # White text
color_border = (0, 0, 255)   # Red border in BGR

is_playing = False

def play_alarm():
    global is_playing
    winsound.Beep(2500, 500) # Frequency 2500Hz, Duration 500ms
    is_playing = False

def play_short_alarm():
    global is_playing
    winsound.Beep(2000, 200) # Short higher-pitch beep
    is_playing = False

print("Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass the image to MediaPipe Face Mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    status = "Open"
    yawning_detected = False
    head_tilt_detected = False
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0] # Pick the primary face
        
        # -------------------------------------------------------------
        # 1. EYE DETECTION (Use eye crops for accurate CNN classification)
        # -------------------------------------------------------------
        # MediaPipe right eye indices: [33, 133, 159, 145]
        # MediaPipe left eye indices: [362, 263, 386, 374]
        left_eye_roi = get_roi_from_landmarks(gray, landmarks, [362, 263, 386, 374], padding=0.5)
        right_eye_roi = get_roi_from_landmarks(gray, landmarks, [33, 133, 159, 145], padding=0.5)
        
        eyes_closed_detected = False
        
        for eye_roi in [left_eye_roi, right_eye_roi]:
            if eye_roi.shape[0] == 0 or eye_roi.shape[1] == 0:
                continue
                
            eye_resized = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE))
            eye_normalized = eye_resized / 255.0  # Rescale as per ImageDataGenerator
            eye_reshaped = np.reshape(eye_normalized, (1, IMG_SIZE, IMG_SIZE, 1))
            
            # Predict the state using the model
            prediction = model.predict(eye_reshaped, verbose=0)
            if np.argmax(prediction) == 1: # Index 1 corresponds to 'Drowsy_Eyes_Closed'
                eyes_closed_detected = True
        
        if eyes_closed_detected:
            status = "Closed"

        # -------------------------------------------------------------
        # 2. MOUTH/YAWN DETECTION (Use face crop for CNN classification)
        # -------------------------------------------------------------
        h_frame, w_frame = frame.shape[:2]
        x_min = int(min([lm.x for lm in landmarks.landmark]) * w_frame)
        x_max = int(max([lm.x for lm in landmarks.landmark]) * w_frame)
        y_min = int(min([lm.y for lm in landmarks.landmark]) * h_frame)
        y_max = int(max([lm.y for lm in landmarks.landmark]) * h_frame)
        
        # Add padding to approximate the broader dlib face bounding box
        pad_w = int((x_max - x_min) * 0.15)
        pad_h = int((y_max - y_min) * 0.20)
        
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(frame.shape[1], x_max + pad_w)
        y2 = min(frame.shape[0], y_max + pad_h)
        
        face_roi = gray[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, IMG_SIZE, IMG_SIZE, 1))
            
            face_pred = model.predict(face_reshaped, verbose=0)
            if np.argmax(face_pred) == 2: # Index 2 corresponds to 'Drowsy_Yawning'
                status = "Yawning"
                yawning_detected = True

        # -------------------------------------------------------------
        # 3. HEAD TILT DETECTION 
        # -------------------------------------------------------------
        angle = calculate_tilt(landmarks, frame)
        if abs(angle) > 20: # 20 degrees threshold
            head_tilt_detected = True
    

    if status == "Closed" or status == "Yawning":
        score += 1
    else:
        score -= 1
        
    if score < 0:
        score = 0
        
    # Draw thicker Red border exactly like reference if passed threshold
    if score >= THRESHOLD:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color_border, 20)
        
       
        if not is_playing:
            is_playing = True
            t = threading.Thread(target=play_alarm)
            t.daemon = True
            t.start()
            
    
    if yawning_detected or head_tilt_detected:
        if not is_playing:
            is_playing = True
            t = threading.Thread(target=play_short_alarm)
            t.daemon = True
            t.start()
            
    # Put Text matching the reference
    if status == "Open":
        text = f"Open   Score: {score}"
    elif status == "Yawning":
        text = f"Yawn   Score: {score}"
    else:
        text = f"Closed Score: {score}"
        
    cv2.putText(frame, text, (15, frame.shape[0] - 20), font, font_scale, color_text, thickness, cv2.LINE_AA)

    # Dynamic UI Overlay for behaviors
    y_pos = 35
    if yawning_detected:
        cv2.putText(frame, "YAWNING DETECTED", (15, y_pos), font, 0.8, (0, 165, 255), 2, cv2.LINE_AA) # Orange text
        y_pos += 35
    
    if head_tilt_detected:
        cv2.putText(frame, "HEAD TILT DETECTED", (15, y_pos), font, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
