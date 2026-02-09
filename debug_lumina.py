import os
import sys
import socket
import cv2
import requests # Needed for Ollama check

def check_socket(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False

def check_ollama():
    try:
        res = requests.get('http://localhost:11434')
        return res.status_code == 200
    except:
        return False

def system_doctor():
    print("\n🏥 SYSTEM DOCTOR 🏥")
    print("----------------------------")
    
    print(f"1. Python Version: {sys.version.split()[0]} ... ", end="")
    if sys.version_info < (3, 8): print("❌ (Requires 3.8+)")
    else: print("✅")

    print("2. Checking Libraries ... ", end="")
    try:
        import qdrant_client
        import ultralytics
        import speech_recognition
        import numpy
        import ollama
        print("✅")
    except ImportError as e:
        print(f"❌ MISSING: {e.name}")
        return

    print("3. Connecting to Qdrant (localhost:6333) ... ", end="")
    if check_socket("localhost", 6333):
        print("✅")
    else:
        print("❌ FAILED (Is Docker running?)")

    print("4. Checking Ollama (localhost:11434) ... ", end="")
    if check_ollama():
        print("✅")
    else:
        print("❌ FAILED (Is Ollama running? Run 'ollama serve')")

    print("5. Checking Model Files ... ", end="")
    if os.path.exists("yolov8m.pt"): print("✅ yolov8m.pt found")
    elif os.path.exists("yolov8n.pt"): print("✅ yolov8n.pt found")
    else: print("❌ No YOLO model found")

if __name__ == "__main__":
    system_doctor()
