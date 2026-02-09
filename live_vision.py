import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from models import SmoothingBuffer, SpatialMath

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
DARK = (15, 15, 15)

class CameraStream:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
    def start(self):
        self.started = True
        threading.Thread(target=self.update, daemon=True).start()
        return self
    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01) 
    def read(self):
        with self.read_lock: return self.frame.copy() if self.grabbed else None
    def stop(self):
        self.started = False
        self.cap.release()

class VisionCore:
    def __init__(self, app, source=0):
        self.app = app
        self.model = YOLO("yolov8n.pt") 
        self.stream = CameraStream(source)
        self.buffers = {} 
        self.prev_gray = None
        self.ALLOWED = [
            "person", "backpack", "bottle", "cup", "chair", "couch", "potted plant", "bed", 
            "tv", "laptop", "mouse", "keyboard", "cell phone", "book"
        ]

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def calculate_optical_flow(self, frame):
        # Basic flow to update heading
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return
        
        # Center crop flow
        h, w = gray.shape
        cy, cx = h//2, w//2
        curr = gray[cy-100:cy+100, cx-100:cx+100]
        prev = self.prev_gray[cy-100:cy+100, cx-100:cx+100]
        
        if curr.shape == prev.shape:
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            shift = np.mean(flow[..., 0])
            # Sensitivity adjustment
            if abs(shift) > 0.5: 
                self.app.world_state.heading -= (shift * 0.3)
        
        self.prev_gray = gray

    def draw_sliding_compass(self, frame, heading):
        h, w = frame.shape[:2]
        
        # 1. Background Tape
        bar_height = 50
        cv2.rectangle(frame, (0, 0), (w, bar_height), DARK, -1)
        
        # 2. Center Indicator (Red Triangle)
        cx = w // 2
        tri_pts = np.array([[cx, bar_height], [cx-10, bar_height-10], [cx+10, bar_height-10]])
        cv2.drawContours(frame, [tri_pts], 0, (0, 0, 255), -1)
        
        # 3. Dynamic Ticks
        # Field of View represented on bar = 100 degrees
        pixels_per_degree = w / 100 
        
        # Iterate from Heading-50 to Heading+50
        start_angle = int(heading) - 50
        end_angle = int(heading) + 51
        
        for angle in range(start_angle, end_angle):
            # Normalize angle to 0-359
            norm_angle = angle % 360
            
            # X position relative to center
            offset = angle - heading
            x_pos = int(cx + (offset * pixels_per_degree))
            
            if 0 <= x_pos <= w:
                # Major Cardinals
                label = ""
                if norm_angle == 0: label = "N"
                elif norm_angle == 45: label = "NE"
                elif norm_angle == 90: label = "E"
                elif norm_angle == 135: label = "SE"
                elif norm_angle == 180: label = "S"
                elif norm_angle == 225: label = "SW"
                elif norm_angle == 270: label = "W"
                elif norm_angle == 315: label = "NW"
                
                if label:
                    cv2.putText(frame, label, (x_pos-10, 30), FONT, 0.8, CYAN, 2)
                    cv2.line(frame, (x_pos, 35), (x_pos, 45), CYAN, 2)
                elif angle % 10 == 0:
                    # Minor ticks
                    cv2.line(frame, (x_pos, 40), (x_pos, 45), (100, 100, 100), 1)

    def run(self):
        self.stream.start()
        print(f"   [VISION] Pipeline Active.")
        
        while True:
            try:
                frame = self.stream.read()
                if frame is None: 
                    time.sleep(0.1)
                    continue
                
                self.calculate_optical_flow(frame)
                
                # Run YOLO if AWAKE
                mode = getattr(self.app, 'mode', 'SLEEP')
                if mode == "AWAKE":
                    results = self.model(frame, verbose=False, conf=0.5)[0]
                    h, w = frame.shape[:2]
                    
                    for box in results.boxes:
                        lbl = self.model.names[int(box.cls[0])]
                        if lbl in self.ALLOWED:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            dist = SpatialMath.estimate_depth(y2-y1, h, lbl)
                            
                            # Angle calculation
                            cx = (x1 + x2) // 2
                            rel_angle = ((cx / w) - 0.5) * 60
                            abs_angle = (self.app.world_state.heading + rel_angle) % 360
                            
                            if self.app.archivist:
                                self.app.archivist.perceive(lbl, abs_angle, dist, conf)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2)
                            cv2.putText(frame, f"{lbl}", (x1, y1-5), FONT, 0.5, CYAN, 1)

                # Draw UI
                self.draw_sliding_compass(frame, self.app.world_state.heading)
                
                # Status
                cv2.putText(frame, f"STATUS: {mode}", (10, frame.shape[0]-10), FONT, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("AI HUD", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Exception as e:
                pass
