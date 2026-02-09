import threading
import queue
from sentence_transformers import SentenceTransformer
from database import MemoryVault

class SmoothingBuffer:
    def __init__(self, window_size=5):
        self.data = []
        self.size = window_size
    def add(self, val):
        self.data.append(val)
        if len(self.data) > self.size: self.data.pop(0)
    def get_average(self):
        if not self.data: return 0.0
        return sum(self.data) / len(self.data)

class WorldState:
    #Tracks User Direction.
    def __init__(self):
        self._heading = 0.0
        self.lock = threading.Lock()
    @property
    def heading(self):
        with self.lock: return self._heading
    @heading.setter
    def heading(self, value):
        with self.lock: self._heading = value % 360

class SpatialMath:
    @staticmethod
    def get_clock_direction(obj_abs, user_abs):
        """
        Returns (clock_position, detailed_text)
        Example: (2, "Turn right to 2 o'clock")
        """
        diff = (obj_abs - user_abs + 180) % 360 - 180
        
        # Calculate Clock Face
        if diff < 0: clock_angle = 360 + diff 
        else: clock_angle = diff
        
        clock_hour = int((clock_angle + 15) // 30)
        if clock_hour == 0: clock_hour = 12
        
        # Generate Text
        dist_deg = abs(diff)
        if dist_deg < 10:
            return 12, "Directly in front of you"
        elif dist_deg > 170:
            return 6, "Directly behind you"
        
        direction = "Right" if diff > 0 else "Left"
        return clock_hour, f"Turn {direction} to {clock_hour} o'clock"

    @staticmethod
    def estimate_depth(bbox_h, frame_h, label):
        # Focal length approx for webcam
        REAL_HEIGHTS = {
            "person": 1.70, "bottle": 0.25, "cup": 0.15, "chair": 1.0, 
            "laptop": 0.30, "cell phone": 0.15, "book": 0.25, "tv": 0.50
        }
        real_h = REAL_HEIGHTS.get(label, 0.5)
        return (real_h * 800) / bbox_h

class CoreSystem:
    """
    The dependency container that holds everything together.
    """
    def __init__(self):
        print("🚀 [CORE] Initializing 5-Agent Architecture...")
        self.world_state = WorldState()
        self.vault = MemoryVault()
        
        # Load embedding model once
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.tts_q = queue.Queue()
        self.status = "SLEEPING" 
        
        # Import agents here to avoid circular imports
        from agents import CoordinatorAgent, ArchivistAgent
        self.coordinator = CoordinatorAgent(self)
        self.archivist = ArchivistAgent(self.vault, self.embedder)
