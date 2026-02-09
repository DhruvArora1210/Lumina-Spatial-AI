--------------------------------------------
**Lumina: Embodied Spatial Intelligence**
--------------------------------------------
"Most AI agents live in chat windows. Lumina lives in the real world."

Lumina is a Multi-Agent Embodied AI designed to assist the visually impaired. Unlike standard RAG systems that retrieve text documents, Lumina retrieves physical location vectors. It grants the AI Object Permanence—the ability to remember where objects are in a room even when the camera is no longer looking at them.
--------------------------------------------
The Architecture: "The Cortex"
--------------------------------------------
Lumina is not a monolith; it is a federation of 5 specialized agents working in an asynchronous loop. This separation of concerns ensures that heavy visual processing does not block cognitive reasoning.

**The 5-Agent Roster**

1.) Coordinator Agent:- Orchestrator
The "Prefrontal Cortex." It parses natural language intent ("I'm thirsty"), plans the workflow, and delegates tasks to other agents. It never touches raw data directly.

2.) Archivist:- Perception
The "Visual Cortex." Runs real-time YOLOv8 detection. It converts 2D pixel bounding boxes into 3D spatial vectors (Distance + Heading) and embeds them into the vector stream.

3.) Librarian:- Memory
The "Hippocampus." Interfaces with Qdrant. It handles Hybrid Search (Semantic lookup for "something to drink" + Strict filtering for "confidence > 0.6").

4.)Janitor:- Maintenance
The "Glia." Prevents memory bloat. It performs spatial deduplication (e.g., "We already saw a bottle at 45° 5 seconds ago, don't write it again").

5.)Critic:- Validation
The "Superego." A deterministic logic layer that validates LLM hallucinations. If the Coordinator says "Found it" but the confidence score is low, the Critic overrides it.
--------------------------------------------
Technical Deep Dive
--------------------------------------------
1. Spatial Vector RAG (The "Secret Sauce")
Standard Vector Search maps Text -> Text. Lumina maps Visual Features -> Spatial Coordinates. When the Archivist sees a "cup", it doesn't just store the word "cup". It stores a payload containing:

Vector: CLIP/All-MiniLM embedding of the label.

Payload: { "angle_abs": 120.5, "dist": 1.2m, "timestamp": 17070500 }.

This allows us to answer questions like "Where did I leave my keys?" by querying the vector space, retrieving the absolute heading stored 10 minutes ago, and calculating the relative turn needed now.

2. Relative Navigation Engine (SpatialMath)
Raw coordinates are useless to a blind user. Lumina's SpatialMath module normalizes the vector space relative to the user's current facing direction.

Formula: (Object_Heading - User_Heading + 180) % 360 - 180

Output: Maps the result to a 12-Sector Clock Face (e.g., "Turn right to 2 O'Clock").

3. Thread-Safe World State
The system runs on two parallel event loops:

Vision Loop (120Hz): Writes to WorldState (Heading/Compass).

Cognitive Loop (On-Demand): Reads from WorldState. We use threading.Lock() mutexes to ensure the "Brain" never acts on stale "Eye" data.
--------------------------------------------
Installation
--------------------------------------------
Prerequisites
Python 3.10+

Qdrant (Running via Docker is recommended)

Ollama (Running llama3 or mistral)

--------------------------------------------
SETUP 
--------------------------------------------
# 1. Clone the repository
git clone https://github.com/DhruvArora1210/Lumina-Spatial-AI.git
cd lumina

# 2. Install dependencies
pip install ultralytics qdrant-client sentence-transformers speechrecognition pyttsx3 ollama opencv-python

# 3. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 4. Start Ollama
ollama run llama3

--------------------------------------------
Running the System
--------------------------------------------
# Verify all systems (Camera, DB, LLM) are active
python debug_lumina.py

# Launch the Core System
python main.py


File Structure Guide
For developers and judges reviewing the code, here is the mental map of the repository:

models.py: Start here. Defines the Data Classes, Math Engines, and the Dependency Injection Container (CoreSystem).

agents.py: The logic layer. Contains the 5 Agent classes and their interaction prompts.

live_vision.py: The "Eyes." Handles OpenCV streaming, YOLO inference, and the HUD overlay.

database.py: The Qdrant wrapper. Handles upsert_spatial and search_spatial_exact.

main.py: The entry point. Initializes the Audio Thread and the Main Event Loop.

 
Why This Matters (The "Social Good" Aspect)
While many Multi-Agent systems focus on optimizing business workflows, Lumina applies Agentic Workflow Patterns to accessibility.

By decoupling Perception (Computer Vision) from Reasoning (LLM), we solve the latency problem. The Vision system can run at 30FPS tracking objects, while the LLM only wakes up when the user asks a question. This makes the system responsive enough for real-time navigation aid.
