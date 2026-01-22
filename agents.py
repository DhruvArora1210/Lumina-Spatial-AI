"""
agents.py - Asynchronous multi-agent system for Lumina v3 PRODUCTION
FIXED: ARKit compatibility with coordinate normalization
"""
import asyncio
from typing import List, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import time
import uuid
import numpy as np

from models import (
    ObjectMemory, Pose6DOF, Coordinates3D, SpatialQueryResult,
    SpatialMath
)
from database import MemoryVault

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WatcherAgent:
    """Asynchronous environmental monitoring agent.
    
    Uses asyncio for non-blocking observation processing.
    Triggers updates only on significant pose changes.
    FIXED: Added ARKit/ARCore coordinate normalization.
    """
    
    MOVEMENT_THRESHOLD = 0.5  # meters
    ROTATION_THRESHOLD = 15.0  # degrees
    HEARTBEAT_INTERVAL = 0.1  # seconds
    
    def __init__(self):
        """Initialize Watcher agent."""
        self.last_capture_pose: Optional[Pose6DOF] = None
        self.current_pose: Optional[Pose6DOF] = None
        self.pending_observations: List[dict] = []
        self.is_running = False
        self.observation_count = 0
        self._lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        logger.info("✓ WatcherAgent initialized")
    
    def _normalize_coordinates(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Transform ARKit coordinates to Lumina coordinate system.
        
        FIXED: Added ARKit/ARCore compatibility.
        
        ARKit/ARCore use right-handed coordinate system:
        - X: Right
        - Y: Up
        - Z: Forward (towards user, so -Z is away/forward in world)
        
        Lumina uses:
        - X: Right
        - Y: Forward
        - Z: Up
        
        Args:
            x: ARKit X coordinate
            y: ARKit Y coordinate
            z: ARKit Z coordinate
            
        Returns:
            Tuple of (lumina_x, lumina_y, lumina_z)
        """
        # Transform ARKit (Right-Handed, Y-up, -Z forward) to Lumina (Y-forward)
        # ARKit: X=right, Y=up, Z=back(-Z is forward)
        # Lumina: X=right, Y=forward, Z=up
        lumina_x = x      # X remains the same (right)
        lumina_y = -z     # ARKit -Z (forward) -> Lumina +Y (forward)
        lumina_z = y      # ARKit Y (up) -> Lumina Z (up)
        
        return lumina_x, lumina_y, lumina_z
    
    async def update_pose(self, new_pose: Pose6DOF, from_arkit: bool = False):
        """Update current pose asynchronously.
        
        Args:
            new_pose: Updated 6DOF pose
            from_arkit: If True, applies ARKit coordinate transformation
        """
        async with self._lock:
            if from_arkit:
                # Apply coordinate transformation
                x, y, z = self._normalize_coordinates(
                    new_pose.coords.x,
                    new_pose.coords.y,
                    new_pose.coords.z
                )
                # Create new pose with normalized coordinates
                self.current_pose = Pose6DOF(
                    coords=Coordinates3D(x=x, y=y, z=z),
                    roll=new_pose.roll,
                    pitch=new_pose.pitch,
                    yaw=new_pose.yaw
                )
            else:
                self.current_pose = new_pose
    
    async def should_capture(self) -> bool:
        """Determine if current pose warrants a new capture.
        
        Returns:
            True if significant movement or rotation detected
        """
        async with self._lock:
            if self.current_pose is None:
                return False
            
            if self.last_capture_pose is None:
                logger.info("🔍 First observation - triggering capture")
                return True
            
            # Check for significant change
            has_changed = self.current_pose.has_significant_change(
                self.last_capture_pose,
                position_threshold=self.MOVEMENT_THRESHOLD,
                rotation_threshold=self.ROTATION_THRESHOLD
            )
            
            if has_changed:
                distance = self.current_pose.coords.distance_to(
                    self.last_capture_pose.coords
                )
                rotation = self.current_pose.rotation_difference(
                    self.last_capture_pose
                )
                logger.info(
                    f"🔍 Significant change detected: "
                    f"Δpos={distance:.2f}m, Δrot={rotation:.1f}°"
                )
            
            return has_changed
    
    async def mark_captured(self, pose: Pose6DOF):
        """Mark pose as captured.
        
        Args:
            pose: Pose where capture occurred
        """
        async with self._lock:
            self.last_capture_pose = pose
            self.observation_count += 1
            logger.info(
                f"✓ Capture #{self.observation_count} marked at "
                f"({pose.coords.x:.2f}, {pose.coords.y:.2f}, {pose.coords.z:.2f})"
            )
    
    async def queue_observation(self, description: str, pose: Pose6DOF):
        """Add observation to processing queue.
        
        Args:
            description: Object description
            pose: Observation pose
        """
        async with self._lock:
            self.pending_observations.append({
                'description': description,
                'pose': pose,
                'timestamp': int(time.time())
            })
    
    async def get_pending_observations(self) -> List[dict]:
        """Retrieve and clear pending observations.
        
        Returns:
            List of pending observations
        """
        async with self._lock:
            observations = self.pending_observations.copy()
            self.pending_observations.clear()
            return observations
    
    async def _background_monitoring_loop(self):
        """Background loop that continuously monitors pose changes.
        
        Runs every HEARTBEAT_INTERVAL seconds to check for significant changes.
        """
        logger.info("🔄 Background monitoring loop started")
        
        while self.is_running:
            try:
                # Check if we should trigger a capture
                should_trigger = await self.should_capture()
                
                if should_trigger:
                    logger.debug("Background loop detected significant pose change")
                
                # Sleep for heartbeat interval
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                
            except asyncio.CancelledError:
                logger.info("Background monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
        
        logger.info("Background monitoring loop exited")
    
    def start_background_loop(self) -> Optional[asyncio.Task]:
        """Start the background monitoring loop.
        
        Creates an asyncio task that continuously polls for pose changes.
        
        Returns:
            asyncio.Task for the background loop
        """
        if self.is_running:
            logger.warning("Background loop already running")
            return self._background_task
        
        self.is_running = True
        self._background_task = asyncio.create_task(self._background_monitoring_loop())
        logger.info("✓ Background monitoring loop task created")
        return self._background_task
    
    async def stop_background_loop(self):
        """Stop the background monitoring loop gracefully."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping background monitoring loop...")
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✓ Background monitoring loop stopped")


class ArchivistAgent:
    """Agent for vectorization and database operations.
    
    Handles embedding generation and conflict resolution.
    """
    
    def __init__(
        self,
        vault: MemoryVault,
        model: SentenceTransformer
    ):
        """Initialize Archivist agent.
        
        Args:
            vault: MemoryVault instance
            model: Pre-loaded SentenceTransformer model instance
        """
        self.vault = vault
        self.model = model
        logger.info(f"✓ ArchivistAgent initialized with shared model")
    
    def _generate_embedding_sync(self, description: str) -> np.ndarray:
        """Synchronous embedding generation for executor.
        
        Args:
            description: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        return self.model.encode(
            description,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def _upsert_memory_sync(self, memory: ObjectMemory, resolve_conflicts: bool) -> bool:
        """Synchronous upsert for executor.
        
        Args:
            memory: ObjectMemory to store
            resolve_conflicts: Whether to check conflicts
            
        Returns:
            Success status
        """
        return self.vault.upsert_memory(memory, resolve_conflicts)
    
    async def archive_observation(
        self,
        description: str,
        coords: Coordinates3D,
        timestamp: int,
        resolve_conflicts: bool = True
    ) -> Optional[str]:
        """Archive an observation asynchronously.
        
        Args:
            description: Object description
            coords: 3D coordinates
            timestamp: Unix timestamp
            resolve_conflicts: Whether to check for conflicts
            
        Returns:
            Object ID if successful, None otherwise
        """
        try:
            # Generate unique ID
            object_id = str(uuid.uuid4())
            
            # Generate embedding (blocking operation in thread pool)
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._generate_embedding_sync,
                description
            )
            
            # Create memory object
            memory = ObjectMemory(
                object_id=object_id,
                object_label=description,
                coords=coords,
                timestamp=timestamp,
                embedding=embedding
            )
            
            # Store in vault (blocking operation in thread pool)
            success = await loop.run_in_executor(
                None,
                self._upsert_memory_sync,
                memory,
                resolve_conflicts
            )
            
            if success:
                logger.info(f"✓ Archived '{description}' [ID: {object_id[:8]}...]")
                return object_id
            else:
                logger.error(f"Failed to archive '{description}'")
                return None
            
        except Exception as e:
            logger.error(f"Archive operation failed: {e}")
            return None
    
    async def batch_archive(
        self,
        observations: List[dict]
    ) -> List[Optional[str]]:
        """Archive multiple observations concurrently.
        
        Args:
            observations: List of observation dicts
            
        Returns:
            List of object IDs (None for failures)
        """
        tasks = [
            self.archive_observation(
                obs['description'],
                obs['pose'].coords,
                obs['timestamp']
            )
            for obs in observations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r if not isinstance(r, Exception) else None for r in results]


class LibrarianAgent:
    """Agent for spatial memory search and retrieval."""
    
    MIN_SIMILARITY = 0.50
    DEFAULT_RADIUS = 5.0
    
    def __init__(self, vault: MemoryVault, model: SentenceTransformer):
        """Initialize Librarian agent.
        
        Args:
            vault: MemoryVault instance
            model: Pre-loaded SentenceTransformer model instance
        """
        self.vault = vault
        self.model = model
        logger.info("✓ LibrarianAgent initialized with shared model")
    
    def _generate_query_embedding_sync(self, query_text: str) -> np.ndarray:
        """Synchronous query embedding for executor.
        
        Args:
            query_text: Query text
            
        Returns:
            Normalized embedding vector
        """
        return self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def _hybrid_search_sync(
        self,
        query_embedding: np.ndarray,
        user_coords: Coordinates3D,
        radius_meters: float,
        min_similarity: float,
        top_k: int
    ) -> List[dict]:
        """Synchronous hybrid search for executor.
        
        Args:
            query_embedding: Query vector
            user_coords: User position
            radius_meters: Search radius
            min_similarity: Minimum similarity
            top_k: Max results
            
        Returns:
            Search results
        """
        return self.vault.hybrid_search(
            query_embedding=query_embedding,
            user_coords=user_coords,
            radius_meters=radius_meters,
            min_similarity=min_similarity,
            top_k=top_k
        )
    
    async def search_memory(
        self,
        query_text: str,
        user_pose: Pose6DOF,
        radius_meters: float = None,
        min_similarity: float = None,
        top_k: int = 3
    ) -> List[dict]:
        """Search spatial memory asynchronously.
        
        Args:
            query_text: Natural language query
            user_pose: User's current pose
            radius_meters: Maximum search radius
            min_similarity: Minimum similarity threshold
            top_k: Maximum results
            
        Returns:
            List of search results
        """
        if radius_meters is None:
            radius_meters = self.DEFAULT_RADIUS
        
        if min_similarity is None:
            min_similarity = self.MIN_SIMILARITY
        
        try:
            logger.info(
                f"🔍 Searching for: '{query_text}' from position "
                f"({user_pose.coords.x:.2f}, {user_pose.coords.y:.2f}, "
                f"{user_pose.coords.z:.2f})"
            )
            
            # Generate query embedding (blocking in thread pool)
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                self._generate_query_embedding_sync,
                query_text
            )
            
            # Perform hybrid search (blocking in thread pool)
            results = await loop.run_in_executor(
                None,
                self._hybrid_search_sync,
                query_embedding,
                user_pose.coords,
                radius_meters,
                min_similarity,
                top_k
            )
            
            logger.info(
                f"✓ Found {len(results)} matching objects "
                f"(threshold: {min_similarity:.2f})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


class GuideAgent:
    """Agent for generating navigation guidance."""
    
    def __init__(self):
        """Initialize Guide agent."""
        logger.info("✓ GuideAgent initialized")
    
    async def generate_guidance(
        self,
        search_results: List[dict],
        user_pose: Pose6DOF
    ) -> List[SpatialQueryResult]:
        """Generate navigation guidance from search results.
        
        Args:
            search_results: Results from LibrarianAgent
            user_pose: User's current pose
            
        Returns:
            List of SpatialQueryResult with guidance
        """
        guidance_list = []
        
        for result in search_results:
            try:
                # Calculate navigation metrics
                distance_3d, distance_horiz, bearing, elevation, clock_pos = \
                    SpatialMath.calculate_navigation_metrics(
                        user_pose,
                        result['coords']
                    )
                
                # Create guidance result
                guidance = SpatialQueryResult(
                    object_id=result['id'],
                    object_label=result['payload']['object_label'],
                    coords=result['coords'],
                    similarity_score=result['score'],
                    distance=distance_3d,
                    horizontal_distance=distance_horiz,
                    bearing=bearing,
                    elevation=elevation,
                    clock_position=clock_pos
                )
                
                guidance_list.append(guidance)
                
                logger.info(
                    f"📍 '{guidance.object_label}': "
                    f"{guidance.clock_position.value}, "
                    f"{guidance.horizontal_distance:.1f}m away, "
                    f"confidence {int(guidance.similarity_score * 100)}%"
                )
                
            except Exception as e:
                logger.error(f"Failed to generate guidance for result: {e}")
                continue
        
        return guidance_list
    
    def get_best_match(
        self,
        guidance_results: List[SpatialQueryResult]
    ) -> Optional[SpatialQueryResult]:
        """Get highest confidence match.
        
        Args:
            guidance_results: List of guidance results
            
        Returns:
            Best match or None
        """
        if not guidance_results:
            return None
        
        # Results are already sorted by similarity
        return guidance_results[0]
    
    async def provide_guidance(
        self,
        query: str,
        user_pose: Pose6DOF,
        librarian: LibrarianAgent,
        radius: float = 5.0
    ) -> Optional[str]:
        """Complete guidance pipeline.
        
        Args:
            query: User's natural language query
            user_pose: User's current pose
            librarian: LibrarianAgent instance
            radius: Search radius
            
        Returns:
            Guidance message or None
        """
        # Search memory
        results = await librarian.search_memory(
            query_text=query,
            user_pose=user_pose,
            radius_meters=radius
        )
        
        if not results:
            return None
        
        # Generate guidance
        guidance_list = await self.generate_guidance(results, user_pose)
        
        if not guidance_list:
            return None
        
        # Get best match
        best = self.get_best_match(guidance_list)
        
        if best:
            return best.guidance_message()
        
        return None