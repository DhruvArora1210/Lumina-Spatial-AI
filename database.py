import time
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, Range

logging.getLogger("httpx").setLevel(logging.WARNING)

class MemoryVault:
    SPATIAL_COL = "lumina_spatial_v10"
    CONTEXT_COL = "lumina_context_v1"
    VECTOR_SIZE = 384 
    
    def __init__(self, host="localhost", port=6333):
        print(f"🔌 [DB] Connecting to Qdrant Core ({host}:{port})...")
        try:
            self.client = QdrantClient(host=host, port=port, timeout=3.0)
            self._setup_collections()
            print("✅ [DB] Qdrant Connected.")
        except Exception as e:
            print(f"❌ [DB] Connection Failed: {e}")
            self.client = None

    def _setup_collections(self):
        if not self.client: return
        if not self.client.collection_exists(self.SPATIAL_COL):
            self.client.create_collection(self.SPATIAL_COL, VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE))

    def search_spatial_exact(self, label):
        if not self.client: return []
        try:
            f = Filter(must=[FieldCondition(key="label", match=MatchValue(value=label))])
            res = self.client.scroll(self.SPATIAL_COL, scroll_filter=f, limit=10, with_payload=True)[0]
            # Sort by newest first
            data = [{"id": p.id, **p.payload} for p in res]
            return sorted(data, key=lambda x: x['timestamp'], reverse=True)
        except: return []

    def search_spatial_semantic(self, vector, limit=3):
        if not self.client: return []
        try:
            hits = self.client.search(self.SPATIAL_COL, query_vector=vector, limit=limit)
            return [{"id": h.id, "score": h.score, **h.payload} for h in hits]
        except: return []

    def upsert_spatial(self, pid, vector, payload):
        if not self.client: return
        self.client.upsert(self.SPATIAL_COL, [PointStruct(id=str(pid), vector=vector, payload=payload)])

    def get_recent_unique_labels(self, seconds=300):
        """Inventory Count: Scans last X seconds for unique objects"""
        if not self.client: return {}
        try:
            cutoff = time.time() - seconds
            f = Filter(must=[FieldCondition(key="timestamp", range=Range(gte=cutoff))])
            # Scroll a batch (approx 50 items)
            res = self.client.scroll(self.SPATIAL_COL, scroll_filter=f, limit=50, with_payload=True)[0]
            
            # Count unique labels
            counts = {}
            for p in res:
                lbl = p.payload['label']
                counts[lbl] = counts.get(lbl, 0) + 1
            
            # Simple heuristic: Div by 5 to estimate actual unique objects (since we log multiple times)
            # Or just return the existence:
            final_counts = {}
            for k in counts:
                # Just say we have "some" or "at least one", or try to estimate
                final_counts[k] = 1 # Simply reporting existence for now
            return final_counts
        except Exception as e:
            print(e)
            return {}
