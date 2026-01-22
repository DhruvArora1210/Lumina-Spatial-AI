👁️ Lumina: Multi-Agent Spatial Intelligence
Lumina is an assistive AI system designed for the visually impaired. It transforms real-time computer vision into a persistent Spatial Knowledge Graph, allowing users to "search" their physical environment using natural language and receive intuitive clock-face directional guidance.


1. Prerequisites
-Python 3.9+

-Docker (to run the Qdrant engine)

-Webcam & Microphone

2. Installation
Bash

# Clone the repository
git clone <https://github.com/DhruvArora1210/Lumina-Spatial-AI>
cd Lumina-Spatial-AI

# Install dependencies



3. Running the System
-Start Qdrant Engine:

Bash

docker run -p 6333:6333 qdrant/qdrant
-Launch Lumina:

Bash

python live_vision.py


🛠️ System Architecture
Lumina operates using a specialized Multi-Agent Orchestration framework:

WatcherAgent: Normalizes raw camera coordinates into a standard 3D spatial frame.

ArchivistAgent: Processes detections, generates CLIP ViT-B/32 embeddings, and manages memory state in Qdrant.

LibrarianAgent: Performs Hybrid Search (Vector + Metadata filtering) to retrieve the most relevant object memories.

GuideAgent: Translates stored 3D vectors into human-centric clock-face directions (e.g., "11 o'clock").

🧠 Qdrant Integration (Technical Depth)
Lumina leverages Qdrant not just for storage, but for Spatial Reasoning:

Multimodal Embeddings: We store 512-dimensional vectors generated from visual labels, allowing the system to understand that a "cell phone" and "mobile" are the same object.

Conflict Resolution: Our MemoryVault uses exact string matching to prevent data collisions, ensuring that a "Cat" doesn't overwrite a "Car" in memory.

Persistence: Unlike standard object detectors, Lumina remembers where an object was even after it leaves the camera's field of view.

🎮 Controls
-S Key: Archive the current scene into Qdrant memory.

-V Key: Trigger voice search (e.g., "Where is my laptop?").

-T Key: Manual text search in the terminal.

-Q Key: Safely shut down the system.
