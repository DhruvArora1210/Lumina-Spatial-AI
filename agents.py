import time
import uuid
import json
import ollama 
from models import SpatialMath

#CONFIG
VISION_LABELS = [
    "person", "backpack", "bottle", "cup", "chair", "couch", "potted plant", "bed", 
    "dining table", "tv", "laptop", "mouse", "keyboard", "cell phone", "book", 
    "clock", "vase", "scissors", "teddy bear", "toothbrush"
]

class JanitorAgent:
    def __init__(self, vault):
        self.vault = vault
        self.cache = {} 

    def clean_and_store(self, label, angle_abs, dist, vec, conf):
        if conf < 0.60: return 
        now = time.time()
        
        # RAM Check
        if label in self.cache:
            last = self.cache[label]
            time_diff = now - last['time']
            ang_diff = abs(last['angle'] - angle_abs)
            if ang_diff > 180: ang_diff = 360 - ang_diff 
            dist_diff = abs(last['dist'] - dist)
            
            # If seen < 5s ago AND moved < 5 deg AND < 1m... Ignore.
            if time_diff < 5.0 and ang_diff < 5.0 and dist_diff < 1.0:
                return 

        print(f"   [JANITOR] 💾 Mapped: '{label}' at {int(angle_abs)}°")
        self.cache[label] = {'angle': angle_abs, 'dist': dist, 'time': now}
        
        db_id = str(uuid.uuid4())
        payload = {
            "label": label, "angle_abs": float(angle_abs), 
            "dist": float(dist), "timestamp": now, "confidence": float(conf)
        }
        self.vault.upsert_spatial(db_id, vec, payload)

class ArchivistAgent:
    #The Observer
    def __init__(self, vault, embedder):
        self.janitor = JanitorAgent(vault)
        self.embedder = embedder

    def perceive(self, label, angle_abs, dist, conf):
        vec = self.embedder.encode(label).tolist()
        self.janitor.clean_and_store(label, angle_abs, dist, vec, conf)

class LibrarianAgent:
    """AGENT 3: The Retriever"""
    def __init__(self, vault, embedder):
        self.vault = vault
        self.embedder = embedder

    def retrieve(self, target_object):
        print(f"   [LIBRARIAN] 🔍 Searching for '{target_object}'...")
        results = self.vault.search_spatial_exact(target_object)
        if results: 
            print(f"   [LIBRARIAN] ✅ Exact match found ({len(results)} records).")
            return results[0], 1.0
        
        vec = self.embedder.encode(target_object).tolist()
        results = self.vault.search_spatial_semantic(vec, limit=1)
        if results: return results[0], results[0]['score']
        return None, 0.0

    def get_inventory(self):
        return self.vault.get_recent_unique_labels(seconds=300)

class CriticAgent:
    #The Evaluator
    def evaluate(self, memory_item, score, target):
        if not memory_item: return False, "I haven't seen that object."
        if score < 0.65: return False, f"I'm not sure if I saw a {target}."
        return True, "Valid"

class CoordinatorAgent:
    """AGENT 5: The Planner"""
    def __init__(self, sys):
        self.sys = sys
        self.librarian = LibrarianAgent(sys.vault, sys.embedder)
        self.critic = CriticAgent()

    def _consult_llm(self, user_text):
        print(f"   [COORDINATOR] 🗣️ User: \"{user_text}\"")
        
        # HIGH-PERFORMANCE PROMPT
        prompt = f"""
        User Command: "{user_text}"
        Vision Labels: {', '.join(VISION_LABELS)}
        
        Task: Map command to JSON.
        
        Rules:
        1. "Where is my bottle" -> intent: FIND, target: bottle
        2. "Bottle" -> intent: FIND, target: bottle
        3. "I need water" -> intent: FIND, target: bottle
        4. "I need to work" -> intent: FIND, target: laptop
        5. "What do you see?" -> intent: INVENTORY
        6. "Bye" or "Stop" -> intent: SLEEP
        
        Output JSON ONLY: {{"intent": "...", "target": "..."}}
        """
        try:
            res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}], format='json')
            return json.loads(res['message']['content'])
        except:
            return {"intent": "UNKNOWN", "target": None}

    def run_workflow(self, command):
        decision = self._consult_llm(command)
        intent = decision.get("intent")
        target = decision.get("target")
        
        print(f"   [COORDINATOR] 📋 Plan: {intent} -> {target}")

        if intent == "SLEEP":
            return {"speech": "Going to sleep.", "status": "SLEEP"}

        elif intent == "INVENTORY":
            counts = self.librarian.get_inventory()
            if not counts: return {"speech": "I see nothing right now.", "status": "AWAKE"}
            report = ", ".join([f"{k}" for k in counts.keys()])
            return {"speech": f"I can see: {report}.", "status": "AWAKE"}
            
        elif intent == "FIND" and target:
            memory, score = self.librarian.retrieve(target)
            is_valid, msg = self.critic.evaluate(memory, score, target)
            
            if is_valid:
                # --- NAVIGATION MATH ---
                user_hdg = self.sys.world_state.heading
                obj_hdg = memory.get('angle_abs', 0)
                dist = memory.get('dist', 0)
                
                clock, guide_text = SpatialMath.get_clock_direction(obj_hdg, user_hdg)
                
                # --- TERMINAL LOG FOR NAV ---
                print(f"   [NAV-LOG] 🧭 User Hdg: {int(user_hdg)}°")
                print(f"   [NAV-LOG] 📍 Obj Hdg:  {int(obj_hdg)}°")
                print(f"   [NAV-LOG] 🕒 Result:   {clock} O'Clock ({guide_text})")
                print(f"   [NAV-LOG] 📏 Dist:     {dist:.2f}m")
                
                speech = f"Found {memory['label']}, {guide_text}, {dist:.2f} meters away."
                return {"speech": speech, "status": "AWAKE"}
            else:
                return {"speech": msg, "status": "AWAKE"}

        return {"speech": "I didn't understand that command.", "status": "AWAKE"}
