import threading
import time
import pyttsx3
import speech_recognition as sr
from models import CoreSystem, WorldState
from live_vision import VisionCore

# --- CONFIG ---
print("\n--- SYSTEM BOOT ---")
cam_source = input("Enter Camera URL (or Press Enter): ").strip()
SOURCE = cam_source if cam_source else 0

app = CoreSystem()
is_speaking = threading.Event()
app.mode = "SLEEPING"

def speech_worker(q):
    engine = pyttsx3.init()
    while True:
        text = q.get()
        if text:
            is_speaking.set()
            try:
                engine.say(text)
                engine.runAndWait()
            except: pass
            is_speaking.clear()

def main_loop():
    threading.Thread(target=speech_worker, args=(app.tts_q,), daemon=True).start()
    vision = VisionCore(app, source=SOURCE)
    vision.start()

    r = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print("🎧 [AUDIO] Calibrating...")
        r.adjust_for_ambient_noise(source, duration=1)
        r.energy_threshold = 250 # More sensitive
        r.pause_threshold = 0.8  # Faster end-of-speech detection

    print("\n💤 SYSTEM ASLEEP. Say 'HELLO' to wake.\n")

    while True:
        try:
            if is_speaking.is_set(): 
                time.sleep(0.5)
                continue

            with mic as source:
                if app.mode == "SLEEPING":
                    try: audio = r.listen(source, timeout=1.0, phrase_time_limit=2.0)
                    except sr.WaitTimeoutError: continue
                else:
                    print(f"   [LISTENING] Window Open (10s)...")
                    try: audio = r.listen(source, timeout=10.0, phrase_time_limit=5.0)
                    except sr.WaitTimeoutError: 
                        print("   [TIMEOUT] No command.")
                        continue

            try:
                text = r.recognize_google(audio).lower()
            except: continue

            if app.mode == "SLEEPING":
                if "hello" in text or "wake" in text:
                    print(f"⚡ [WAKE]: '{text}'")
                    app.tts_q.put("Online.")
                    app.mode = "AWAKE"
                continue

            if app.mode == "AWAKE":
                if "bye" in text or "sleep" in text:
                    print(f"💤 [SLEEP]: '{text}'")
                    app.tts_q.put("Offline.")
                    app.mode = "SLEEPING"
                    continue

                print(f"🎤 [CMD]: '{text}'")
                res = app.coordinator.run_workflow(text)
                if res["speech"]: app.tts_q.put(res["speech"])
                if res.get("status") == "SLEEP": app.mode = "SLEEPING"

        except KeyboardInterrupt: break
        except Exception: pass

if __name__ == "__main__":
    main_loop()
