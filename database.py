"""
database.py - Memory Vault with strict string comparison
FIXED: Cat vs. Car data loss - removed Levenshtein, using exact string match
"""
from typing import List, Optional, Dict
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range
)
from qdrant_client.http import models
import numpy as np
import time

from models import ObjectMemory, Coordinates3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryVaultError(Exception):
    """Custom exception for Memory Vault operations."""
    pass


class MemoryVault:
    """Advanced spatial memory database with state management.
    
    Handles object lifecycle, conflict resolution, and stale data detection.
    """
    
    COLLECTION_NAME = "spatial_memory"
    VECTOR_SIZE = 512  # CLIP ViT-B/32
    SIMILARITY_THRESHOLD = 0.65
    SPATIAL_RADIUS_DEFAULT = 5.0  # meters
    POSITION_CHANGE_THRESHOLD = 0.3  # meters for "same location"
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize Memory Vault with Qdrant connection.
        
        Args:
            host: Qdrant server hostname
            port: Qdrant server port
            
        Raises:
            MemoryVaultError: If connection fails
        """
        try:
            self.client = QdrantClient(host=host, port=port, timeout=10.0)
            self._initialize_collection()
            logger.info(f"✓ Memory Vault connected to Qdrant at {host}:{port}")
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant: {e}"
            logger.error(error_msg)
            raise MemoryVaultError(error_msg)
    
    def _initialize_collection(self):
        """Create collection with optimized HNSW indexing."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Number of connections per node
                        ef_construct=200,  # Construction time quality
                        full_scan_threshold=10000
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000
                    )
                )
                logger.info(f"✓ Created collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"✓ Collection '{self.COLLECTION_NAME}' exists")
        except Exception as e:
            raise MemoryVaultError(f"Collection initialization failed: {e}")
    
    def check_object_at_location(
        self,
        coords: Coordinates3D,
        threshold: float = None
    ) -> Optional[Dict]:
        """Check if an object exists at or near given coordinates.
        
        Used for state management to detect moved/removed objects.
        
        Args:
            coords: Coordinates to check
            threshold: Distance threshold for "same location"
            
        Returns:
            Object data if found, None otherwise
        """
        if threshold is None:
            threshold = self.POSITION_CHANGE_THRESHOLD
        
        try:
            # Search for objects near this location
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key='x',
                        range=Range(
                            gte=coords.x - threshold,
                            lte=coords.x + threshold
                        )
                    ),
                    FieldCondition(
                        key='y',
                        range=Range(
                            gte=coords.y - threshold,
                            lte=coords.y + threshold
                        )
                    ),
                    FieldCondition(
                        key='z',
                        range=Range(
                            gte=coords.z - threshold,
                            lte=coords.z + threshold
                        )
                    )
                ]
            )
            
            # Scroll through nearby points
            results, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                return None
            
            # Find closest object
            closest = None
            min_distance = float('inf')
            
            for result in results:
                obj_coords = Coordinates3D(
                    x=result.payload['x'],
                    y=result.payload['y'],
                    z=result.payload['z']
                )
                distance = coords.distance_to(obj_coords)
                
                if distance < min_distance:
                    min_distance = distance
                    closest = {
                        'id': result.id,
                        'payload': result.payload,
                        'distance': distance
                    }
            
            return closest if min_distance < threshold else None
            
        except Exception as e:
            logger.error(f"Location check failed: {e}")
            return None
    
    def upsert_memory(
        self,
        memory: ObjectMemory,
        resolve_conflicts: bool = True
    ) -> bool:
        """Insert or update object memory with strict string comparison.
        
        FIXED: Removed Levenshtein distance - now uses exact string match only.
        This prevents "Cat" from overwriting "Car".
        
        Args:
            memory: ObjectMemory to store
            resolve_conflicts: Whether to check for conflicts at location
            
        Returns:
            True if successful
        """
        if memory.embedding is None:
            logger.error("Cannot upsert memory without embedding")
            return False
        
        try:
            # Check for conflicts if requested
            if resolve_conflicts:
                existing = self.check_object_at_location(memory.coords)
                
                if existing and existing['id'] != memory.object_id:
                    old_label = existing['payload']['object_label']
                    new_label = memory.object_label
                    
                    # FIXED: Strict string comparison only
                    # Delete old object only if labels match exactly (case-insensitive)
                    is_same_object = new_label.lower() == old_label.lower()
                    
                    if is_same_object:
                        logger.warning(
                            f"Same object detected (exact match): "
                            f"Deleting '{old_label}' and updating with '{new_label}'"
                        )
                        self.client.delete(
                            collection_name=self.COLLECTION_NAME,
                            points_selector=models.PointIdsList(
                                points=[existing['id']]
                            )
                        )
                    else:
                        logger.info(
                            f"Different objects at nearby locations: "
                            f"Keeping '{old_label}' and adding '{new_label}'"
                        )
            
            # Create point for upsert
            point = PointStruct(
                id=memory.object_id,
                vector=memory.embedding.tolist(),
                payload=memory.to_payload()
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            
            logger.info(
                f"✓ Stored '{memory.object_label}' at "
                f"({memory.coords.x:.2f}, {memory.coords.y:.2f}, {memory.coords.z:.2f})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert memory: {e}")
            return False
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        user_coords: Coordinates3D,
        radius_meters: float = None,
        min_similarity: float = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Perform hybrid spatial-semantic search with filtering.
        
        Args:
            query_embedding: 512-dim query vector
            user_coords: User's current coordinates
            radius_meters: Maximum search radius
            min_similarity: Minimum similarity threshold
            top_k: Maximum results to return
            
        Returns:
            List of search results meeting criteria
        """
        if radius_meters is None:
            radius_meters = self.SPATIAL_RADIUS_DEFAULT
        
        if min_similarity is None:
            min_similarity = self.SIMILARITY_THRESHOLD
        
        try:
            # Spatial filter using bounding box
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key='x',
                        range=Range(
                            gte=user_coords.x - radius_meters,
                            lte=user_coords.x + radius_meters
                        )
                    ),
                    FieldCondition(
                        key='y',
                        range=Range(
                            gte=user_coords.y - radius_meters,
                            lte=user_coords.y + radius_meters
                        )
                    ),
                    FieldCondition(
                        key='z',
                        range=Range(
                            gte=user_coords.z - radius_meters,
                            lte=user_coords.z + radius_meters
                        )
                    )
                ]
            )
            
            # Perform vector search
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=top_k * 3,  # Get extra for filtering
                with_payload=True,
                score_threshold=min_similarity
            )
            
            # Post-process: accurate distance filtering and scoring
            filtered_results = []
            
            for result in results:
                obj_coords = Coordinates3D(
                    x=result.payload['x'],
                    y=result.payload['y'],
                    z=result.payload['z']
                )
                
                distance = user_coords.distance_to(obj_coords)
                
                # Apply strict spatial filter
                if distance <= radius_meters and result.score >= min_similarity:
                    filtered_results.append({
                        'id': result.id,
                        'score': result.score,
                        'payload': result.payload,
                        'distance': distance,
                        'coords': obj_coords
                    })
            
            # Sort by similarity score (descending)
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(
                f"Found {len(filtered_results)} objects "
                f"(similarity ≥ {min_similarity:.2f}, radius ≤ {radius_meters}m)"
            )
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def cleanup_stale_memories(
        self,
        max_age_seconds: int = 3600,
        current_time: int = None
    ) -> int:
        """Remove stale object memories from database.
        
        Args:
            max_age_seconds: Age threshold for removal
            current_time: Current timestamp (default: now)
            
        Returns:
            Number of memories removed
        """
        if current_time is None:
            current_time = int(time.time())
        
        try:
            # Get all memories
            results, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            stale_ids = []
            
            for result in results:
                last_seen = result.payload.get('last_seen', result.payload['timestamp'])
                age = current_time - last_seen
                
                if age > max_age_seconds:
                    stale_ids.append(result.id)
            
            if stale_ids:
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=stale_ids)
                )
                logger.info(f"✓ Cleaned up {len(stale_ids)} stale memories")
            
            return len(stale_ids)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_all_memories(self) -> List[Dict]:
        """Retrieve all stored memories for debugging.
        
        Returns:
            List of all memory records
        """
        try:
            results, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    'id': r.id,
                    'label': r.payload['object_label'],
                    'coords': (r.payload['x'], r.payload['y'], r.payload['z']),
                    'timestamp': r.payload['timestamp']
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def clear_all_memories(self):
        """Delete all memories (for testing/reset)."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._initialize_collection()
            logger.info("✓ All memories cleared")
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")