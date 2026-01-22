import cv2
import asyncio
import threading
from ultralytics import YOLO
from models import Pose6DOF, Coordinates3D
from main import LuminaSystem


async def run_lumina_live():
    lumina = LuminaSystem()
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    print("\n--- LUMINA MULTI-MODAL ---")
    print("[S] Save Memory | [V] Voice Search | [T] Text Search | [Q] Quit\n")

    def process_query(query_text):
        """Runs the search and speaking in a separate thread."""
        if query_text:
            print(f"🔍 Searching for: {query_text}")
            temp_loop = asyncio.new_event_loop()
            user_pose = Pose6DOF(coords=Coordinates3D(0, 0, 0), yaw=0)
            guidance = temp_loop.run_until_complete(lumina.locate_object(query_text, user_pose))
            lumina.say(guidance)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, "S: SAVE | V: VOICE | T: TEXT", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Lumina Vision', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        # SAVE MEMORY
        if key == ord('s'):
            labels = list(set([model.names[int(c)] for c in results[0].boxes.cls]))
            if labels:
                print(f"💾 Saving: {labels}")
                for label in labels:
                    asyncio.create_task(lumina.observe_at_location(label, Coordinates3D(0.5, 0.5, 1.0),
                                                                   Pose6DOF(Coordinates3D(0, 0, 0), 0)))

        # VOICE SEARCH
        if key == ord('v'):
            def voice_worker():
                query = lumina.listen()
                if query: process_query(query)

            threading.Thread(target=voice_worker, daemon=True).start()

        # TEXT SEARCH
        if key == ord('t'):
            query = input("\n⌨️ Type your search (e.g., 'Where is the bottle?'): ")
            threading.Thread(target=process_query, args=(query,), daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(run_lumina_live())