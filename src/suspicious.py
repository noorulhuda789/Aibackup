import cv2
import datetime
import os
import pandas as pd
import torch
import winsound
from pynput import keyboard
import pygetwindow as gw
from fer import FER

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define prohibited objects
prohibited_objects = ['cell phone', 'person', 'laptop']

# Directory to save images and logs
save_dir = r"D:\5th semster\ai project\GradeGuard\images"
log_file = r"D:\5th semster\ai project\GradeGuard\suspicious_activity_log.csv"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Load Haar Cascade for head detection
head_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Create a log DataFrame if not exists
if not os.path.exists(log_file):
    log_columns = [
        'Timestamp', 'Prohibited Object', 'Eye Movement Detected', 'Key Pressed',
        'Tab Change', 'Window Focus Changed', 'Suspicious Expression'
    ]
    log_df = pd.DataFrame(columns=log_columns)
    log_df.to_csv(log_file, index=False)


def log_event(data):
    """Logs data to the CSV log file."""
    log_entry = pd.DataFrame([data], columns=[
        'Timestamp', 'Prohibited Object', 'Eye Movement Detected', 'Key Pressed',
        'Tab Change', 'Window Focus Changed', 'Suspicious Expression'
    ])
    log_entry.to_csv(log_file, mode='a', header=False, index=False)


def on_press(key):
    """Handles keyboard press events."""
    try:
        if key.char == 'q':  # Example: Press 'q' to quit
            log_event([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'N/A', 'N/A', 'q', 'N/A', 'N/A', 'N/A'])
    except AttributeError:
        if key == keyboard.Key.tab:
            log_event([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'N/A', 'N/A', 'Tab', 'N/A', 'N/A', 'N/A'])


def log_window_focus_change(last_window):
    """Logs window focus changes."""
    current_window = gw.getActiveWindow()
    if current_window and current_window.title != last_window:
        log_event([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'N/A', 'N/A', 'N/A', 'N/A',
            f"Changed to: {current_window.title}", 'N/A'
        ])
        return current_window.title
    return last_window


# Add a global variable to track persons detected
person_count = 0  # Number of persons detected


def handle_prohibited_objects(detections, frame):
    """Handles detections of prohibited objects, including logging person disappearance."""
    global suspicious_activity_detected, video_writer, person_count

    detected_persons = 0  # Flag for counting detected persons
    for _, row in detections.iterrows():
        obj_name = row['name']
        if obj_name == 'person':  # Check if a person is detected
            detected_persons += 1  # Increase person count
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {detected_persons}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if obj_name in prohibited_objects:  # Handle prohibited objects
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ALERT: {obj_name} detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            log_event([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), obj_name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

            winsound.Beep(1000, 500)

            if not suspicious_activity_detected:
                suspicious_activity_detected = True
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = os.path.join(save_dir, f"suspicious_activity_{timestamp}.avi")
                video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    # Log if more than one person is detected
    if detected_persons > 1:
        log_event([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Multiple Persons Detected', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

    # Update person count
    if detected_persons != person_count:
        person_count = detected_persons


def detect_head_movements(frame, gray_frame):
    """Detects head movements using Haar Cascade."""
    heads = head_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in heads:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Head Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Initialize keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Track window focus
last_window_title = gw.getActiveWindow().title if gw.getActiveWindow() else None
frame_count = 0
suspicious_activity_detected = False
video_writer = None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 == 0:  # Optimize performance by running detection every nth frame
        results = model(frame)
        handle_prohibited_objects(results.pandas().xyxy[0], frame)

    if suspicious_activity_detected and video_writer:
        video_writer.write(frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_head_movements(frame, gray_frame)

    last_window_title = log_window_focus_change(last_window_title)
    cv2.imshow("GradeGuard - Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
