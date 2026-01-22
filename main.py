import asyncio
import time
import os
import wave
import struct
import pyttsx3
import speech_recognition as sr
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import Pose6DOF, Coordinates3D
from database import MemoryVault
from agents import WatcherAgent, ArchivistAgent, LibrarianAgent, GuideAgent


class LuminaSystem:
    def __init__(self, db_host: str = "localhost", db_port: int = 6333):
        self.vault = MemoryVault(host=db_host, port=db_port)
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('clip-ViT-B-32', device='cpu')

        self.watcher = WatcherAgent()
        self.archivist = ArchivistAgent(self.vault, self.embedding_model)
        self.librarian = LibrarianAgent(self.vault, self.embedding_model)
        self.guide = GuideAgent()

        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty('rate', 175)
        self.recognizer = sr.Recognizer()
        print("✅ LUMINA ENGINE: Hybrid Mode (Voice + Text) Online")

    def say(self, text: str):
        """Speak out loud."""
        try:
            print(f"🤖 Lumina says: {text}")
            self.speech_engine.say(text)
            self.speech_engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")

    def listen(self) -> Optional[str]:
        """Stable Mac Listener using PvRecorder."""
        from pvrecorder import PvRecorder
        path = "temp_capture.wav"
        recorder = PvRecorder(device_index=-1, frame_length=512)
        audio_data = []
        try:
            print("👂 Listening (4s)...")
            recorder.start()
            for _ in range(130):
                frame = recorder.read()
                audio_data.extend(frame)
            recorder.stop()
            with wave.open(path, 'w') as f:
                f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
                f.writeframes(struct.pack("h" * len(audio_data), *audio_data))
            with sr.AudioFile(path) as source:
                audio = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio)
        except Exception as e:
            print(f"⚠️ Voice Error: {e}")
            return None
        finally:
            recorder.delete()
            if os.path.exists(path): os.remove(path)

    async def observe_at_location(self, description, object_coords, observer_pose):
        return await self.archivist.archive_observation(description, object_coords, int(time.time()), True)

    async def locate_object(self, query: str, user_pose: Pose6DOF):
        return await self.guide.provide_guidance(query, user_pose, self.librarian)