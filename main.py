import cv2
import time
import numpy as np
import random
from ultralytics import YOLO
import os

# Load Models
# 1. Standard Model (Cars, Trucks, etc.)
general_model = YOLO("yolov8n.pt")

# 2. Custom Ambulance Model (Optional)
# If 'ambulance.pt' exists, we use it. If not, we fall back to manual trigger 'E'.
ambulance_model_path = "ambulance.pt"
has_ambulance_model = os.path.exists(ambulance_model_path)
ambulance_model = None

if has_ambulance_model:
    print("Loading Custom Ambulance Model...")
    ambulance_model = YOLO(ambulance_model_path)
else:
    print("No 'ambulance.pt' found. Using Manual Override (Press 'E') for demonstration.")

class TrafficLane:
    def __init__(self, lane_id, is_real_camera=False):
        self.lane_id = lane_id
        self.is_real_camera = is_real_camera
        self.vehicle_count = 0
        self.emergency_detected = False
        self.detected_objects = []
        self.green_duration = 0
        self.status = "RED" # RED, GREEN, YELLOW
    
    def update_counts(self, frame, gen_model, amb_model):
        """
        Updates vehicle count and emergency status.
        """
        self.vehicle_count = 0
        self.detected_objects = []
        # Emergency status persists for a frame cycle, reset here unless re-detected
        # (For simulation, we interpret 'E' keypress in the controller, not here)
        if self.is_real_camera:
           self.emergency_detected = False # Reset before detection

        if self.is_real_camera and frame is not None:
            # 1. General Detection (Count Vehicles)
            results = gen_model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    
                    # YOLO class IDs: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
                    if class_id in [2, 3, 5, 7]:
                        self.vehicle_count += 1
                        color = (0, 255, 255) # Yellow
                        self.detected_objects.append((x1, y1, x2, y2, color))
                        
            # 2. Ambulance Detection (Custom Model)
            if amb_model:
                amb_results = amb_model(frame, verbose=False)
                for result in amb_results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.5: # Trusted detection
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            self.emergency_detected = True
                            # Draw Red Box for Ambulance
                            self.detected_objects.append((x1, y1, x2, y2, (0, 0, 255)))
                        
        else:
            # SIMULATION LOGIC
            pass 

    def set_exposure_density(self):
        """
        Sets a new random density for simulation when a cycle starts.
        Also simulates a random Emergency Vehicle chance (5%).
        """
        if not self.is_real_camera:
            self.vehicle_count = random.randint(0, 15)
            # 5% chance of ambulance in simulated lanes
            self.emergency_detected = (random.random() < 0.05) 

class TrafficController:
    def __init__(self):
        self.lanes = [
            TrafficLane(1, is_real_camera=True),
            TrafficLane(2),
            TrafficLane(3),
            TrafficLane(4)
        ]
        self.current_lane_idx = 0
        self.last_switch_time = time.time()
        self.is_transitioning = False # State for Yellow Light
        
        # State: Emergency
        self.emergency_active = False 
        self.emergency_lane_idx = -1
        
        # Calibration
        self.min_green_time = 5
        self.max_green_time = 30
        self.time_per_vehicle = 2 
        
        # Initial Setup
        self.activate_lane(0)

    def calculate_green_time(self, vehicle_count):
        """Weighted Round Robin Logic"""
        calculated_time = self.min_green_time + (vehicle_count * self.time_per_vehicle)
        return min(calculated_time, self.max_green_time)

    def activate_lane(self, lane_idx, is_emergency=False):
        # Reset all to RED
        for lane in self.lanes:
            lane.status = "RED"
        
        self.current_lane_idx = lane_idx
        current_lane = self.lanes[self.current_lane_idx]
        
        # Calculate Time
        if is_emergency:
            duration = 15 # Fixed generous time for emergency
            current_lane.green_duration = duration
            self.emergency_active = True
            self.emergency_lane_idx = lane_idx
            print(f"!!! EMERGENCY OVERRIDE: Lane {current_lane.lane_id} !!!")
        else:
            if not current_lane.is_real_camera:
                current_lane.set_exposure_density()
            duration = self.calculate_green_time(current_lane.vehicle_count)
            current_lane.green_duration = duration
            self.emergency_active = False
            print(f"Lane {current_lane.lane_id} Active. Count: {current_lane.vehicle_count}, Duration: {duration}s")
            
        current_lane.status = "GREEN"
        self.last_switch_time = time.time()
        self.is_transitioning = False

    def update(self):
        current_time = time.time()
        current_lane = self.lanes[self.current_lane_idx]
        elapsed = current_time - self.last_switch_time
        
        # CHECK FOR EMERGENCY (Preemption)
        # Scan all lanes for emergency signal
        found_emergency_idx = -1
        for i, lane in enumerate(self.lanes):
            if lane.emergency_detected:
                found_emergency_idx = i
                break
        
        # Logic: If emergency found AND we are not already serving that emergency
        if found_emergency_idx != -1 and not (self.emergency_active and self.emergency_lane_idx == found_emergency_idx):
            # Immediate Override
            # But wait! If we are already Green on that lane, just extend time? 
            # For simplicity: Switch immediately to that lane as an Emergency Activate
            
            # If current lane is NOT the emergency lane, Force Yellow -> Switch
            if found_emergency_idx != self.current_lane_idx:
                # Force switch behavior
                # Ideally, we should do a fast yellow. For demo, we jump to that lane.
                self.activate_lane(found_emergency_idx, is_emergency=True)
                return
            else:
                # If we are already on that lane, just ensure we are in Emergency Mode
                if not self.emergency_active:
                    self.emergency_active = True
                    current_lane.green_duration += 10 # Extend time
                    print("Emergency in current lane! Extending time.")

        if not self.is_transitioning:
            # Green Light Phase
            if elapsed >= current_lane.green_duration:
                # Switch to Yellow
                current_lane.status = "YELLOW"
                self.last_switch_time = current_time
                self.is_transitioning = True
        else:
            # Yellow Light Phase (Fixed 3 seconds)
            if elapsed >= 3.0:
                # Move to next lane (Round Robin)
                # If returning from emergency, just go to next in cycle
                next_idx = (self.current_lane_idx + 1) % 4
                self.activate_lane(next_idx)

    def get_display_info(self):
        current_lane = self.lanes[self.current_lane_idx]
        elapsed = time.time() - self.last_switch_time
        
        if self.is_transitioning:
            remaining = max(0, 3.0 - elapsed)
        else:
            remaining = max(0, current_lane.green_duration - elapsed)
            
        return {
            "active_lane": current_lane.lane_id,
            "status": current_lane.status,
            "remaining_time": int(remaining),
            "lanes": self.lanes,
            "emergency": self.emergency_active
        }

# Main Execution
def main():
    cap = cv2.VideoCapture(0)
    controller = TrafficController()
    
    print("Traffic System Started.")
    print("Press 'E' to toggle Manual Emergency Simulation for Lane 1.")
    print("Press 'ESC' to exit.")

    manual_emergency_trigger = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Key Controls
        key = cv2.waitKey(30) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('e'):
            manual_emergency_trigger = not manual_emergency_trigger
            print(f"Manual Emergency Trigger: {manual_emergency_trigger}")

        # 1. Update Real Lane Processing
        controller.lanes[0].update_counts(frame, general_model, ambulance_model)
        
        # Apply manual trigger if no model
        if not has_ambulance_model and manual_emergency_trigger:
            controller.lanes[0].emergency_detected = True
            # Visual indicator on frame
            cv2.putText(frame, "MANUAL EMERGENCY TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # 2. Update Controller Logic
        controller.update()
        
        # 3. Visualization
        info = controller.get_display_info()
        
        height, width, _ = frame.shape
        panel_width = 350
        ui_canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        ui_canvas[:, :width] = frame 
        ui_canvas[:, width:] = (40, 40, 40) 
        
        # Draw YOLO Boxes
        for (x1, y1, x2, y2, color) in controller.lanes[0].detected_objects:
            cv2.rectangle(ui_canvas, (x1, y1), (x2, y2), color, 2)
            
        # Dashboard
        start_x = width + 20
        y = 40
        
        title_color = (255, 255, 255)
        if info['emergency']:
            title_color = (0, 0, 255)
            cv2.putText(ui_canvas, "!!! EMERGENCY !!!", (start_x, y), cv2.FONT_HERSHEY_BOLD, 0.8, (0, 0, 255), 2)
            y += 40
        else:
            cv2.putText(ui_canvas, "TRAFFIC CONTROL", (start_x, y), cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
            y += 40
        
        for lane in info['lanes']:
            # Lane Header
            l_color = (150, 150, 150)
            status_text = lane.status
            
            if lane.status == "GREEN": l_color = (0, 255, 0)
            elif lane.status == "YELLOW": l_color = (0, 255, 255)
            elif lane.status == "RED": l_color = (0, 0, 255)
            
            # Marker for emergency
            e_mark = " (!)" if lane.emergency_detected else ""
            
            cv2.putText(ui_canvas, f"LANE {lane.lane_id}{e_mark}: {status_text}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
            y += 25
            cv2.putText(ui_canvas, f"  Vehicles: {lane.vehicle_count}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 35
        
        y += 20
        # Timer
        t_color = (255, 255, 255)
        if info['emergency']: t_color = (0, 0, 255)
        cv2.putText(ui_canvas, f"Time: {info['remaining_time']}s", (start_x, y), cv2.FONT_HERSHEY_BOLD, 1.5, t_color, 3)

        cv2.imshow("Smart Traffic System", ui_canvas)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

