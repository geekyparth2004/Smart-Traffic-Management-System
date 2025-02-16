import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # Using a pre-trained YOLOv8 model

# Traffic light state
RED, GREEN, YELLOW = "RED", "GREEN", "YELLOW"
light_state = RED
default_time = 30
extended_time = 60
current_timer = default_time
last_switch_time = time.time()

# Initialize camera
cap = cv2.VideoCapture(0)

# Detect vehicles and humans using YOLO
def detect_objects(frame):
    results = yolo_model(frame)
    vehicle_count = 0
    person_count = 0
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            
            # YOLO class IDs: 0 = person, 2-7 = vehicles (car, bus, truck, motorcycle, etc.)
            if class_id == 0:
                person_count += 1
                color = (255, 0, 0)  # Blue for persons
            elif class_id in [2, 3, 5, 7]:
                vehicle_count += 1
                color = (0, 255, 255)  # Yellow for vehicles
            else:
                continue
            
            detected_objects.append((x1, y1, x2, y2, color))
    
    return vehicle_count, person_count, detected_objects

def process_frame(frame):
    global light_state, current_timer, last_switch_time
    vehicle_count, person_count, detected_objects = detect_objects(frame)
    total_objects = vehicle_count + person_count
    current_time = time.time()
    
    if total_objects > 5:
        if light_state != GREEN:
            light_state = GREEN
            current_timer = extended_time  # Activate extended timer
            last_switch_time = current_time
    elif current_time - last_switch_time >= current_timer:
        if light_state == GREEN:
            light_state = YELLOW
            current_timer = default_time
            last_switch_time = current_time
        elif light_state == YELLOW:
            light_state = RED
            current_timer = default_time
            last_switch_time = current_time
        elif light_state == RED:
            light_state = GREEN
            current_timer = default_time
            last_switch_time = current_time
    
    return frame, vehicle_count, person_count, detected_objects

def display_traffic_light(frame, vehicle_count, person_count, detected_objects):
    time_remaining = max(0, current_timer - int(time.time() - last_switch_time))
    cv2.putText(frame, f"Traffic Light: {light_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Timer: {time_remaining}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Person Count: {person_count}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw bounding boxes around detected objects
    for (x1, y1, x2, y2, color) in detected_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, vehicle_count, person_count, detected_objects = process_frame(frame)
    frame = display_traffic_light(frame, vehicle_count, person_count, detected_objects)
    
    cv2.imshow("Traffic Control", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
