"""
GAE-RAG: Hybrid Vector Store (ChromaDB)
=========================================
Manages TWO ChromaDB collections:
  1. "original_index"        — vectors from raw chunk text
  2. "aspect_enhanced_index" — vectors from chunk + verified aspects

Both collections store the same metadata and can be queried independently
or jointly (score fusion).

WHY ChromaDB?
  - 100% free and open source
  - Runs entirely locally (no cloud, no signup)
  - Persistent storage (data survives restarts)
  - Fast cosine similarity search
  - Supports metadata filtering

INSTALL:
  pip install chromadb
"""

import json
import os
from typing import Optional


class HybridVectorStore:
    """
    Dual-index ChromaDB vector store for GAE-RAG.
    
    Args:
        persist_dir: Directory to store ChromaDB data (default: ./gae_rag_db)
        collection_prefix: Prefix for collection names (useful for multiple experiments)
    
    USAGE:
        store = HybridVectorStore(persist_dir="./my_experiment")
        store.add_chunks(indexed_chunks)        # index all chunks
        results = store.query(query_vectors, k=5)  # retrieve
    """

    def __init__(
        self,
        persist_dir: str = "./gae_rag_db",
        collection_prefix: str = "gae_rag",
    ):
        self.persist_dir = persist_dir
        self.collection_prefix = collection_prefix
        self._client = None
        self._orig_col = None
        self._aspect_col = None

    def _init_db(self):
        """Lazy-initialize ChromaDB client and collections."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.config import Settings

        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        orig_name = f"{self.collection_prefix}_original"
        aspect_name = f"{self.collection_prefix}_aspect_enhanced"

        # get_or_create allows resuming without re-indexing
        self._orig_col = self._client.get_or_create_collection(
            name=orig_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._aspect_col = self._client.get_or_create_collection(
            name=aspect_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"  [VectorStore] Initialized: '{orig_name}' + '{aspect_name}'")
        print(f"  [VectorStore] Persisted at: {os.path.abspath(self.persist_dir)}")

    def add_chunks(self, indexed_chunks: list[dict], batch_size: int = 64):
        """
        Add a list of fully-indexed chunks to both collections.
        
        Each indexed_chunk must have:
          - chunk_id              (str)
          - text                  (str)
          - original_embedding    (list[float])
          - aspect_enhanced_embedding (list[float])
          - aspects               (list[dict])
          - summary               (str)
          - key_topics            (list[str])
          - contrastive_scope     (list[str])
          - doc_id                (str)
          - metadata              (dict, optional)
        
        Args:
            indexed_chunks: List of chunk dicts from the full pipeline
            batch_size: Chunks per ChromaDB upsert call (larger = faster but more memory)
        """
        self._init_db()

        ids, orig_vecs, aspect_vecs, docs, metas = [], [], [], [], []

        for chunk in indexed_chunks:
            chunk_id = chunk["chunk_id"]

            # Serialize complex fields to JSON strings (ChromaDB only stores strings/ints/floats)
            meta = {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "summary": chunk.get("summary", "")[:500],  # truncate for storage
                "key_topics": json.dumps(chunk.get("key_topics", [])),
                "contrastive_scope": json.dumps(chunk.get("contrastive_scope", [])),
                "aspect_count": len(chunk.get("aspects", [])),
                "aspect_text": chunk.get("aspect_text", "")[:1000],
                "has_error": bool(chunk.get("generation_error")),
            }

            # Merge any extra metadata from the source document
            for k, v in chunk.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"src_{k}"] = v

            ids.append(chunk_id)
            orig_vecs.append(chunk["original_embedding"])
            aspect_vecs.append(chunk["aspect_enhanced_embedding"])
            docs.append(chunk["text"])
            metas.append(meta)

        # Batch upsert
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            slice_ids = ids[start:end]

            self._orig_col.upsert(
                ids=slice_ids,
                embeddings=orig_vecs[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )
            self._aspect_col.upsert(
                ids=slice_ids,
                embeddings=aspect_vecs[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )

            print(f"  [VectorStore] Indexed {end}/{total} chunks")

        print(f"  [VectorStore] Done. Total in store: {self._orig_col.count()} chunks")

    def query(
        self,
        query_vectors: dict,
        k: int = 5,
        alpha: float = 0.5,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query both indexes and fuse results using score fusion.
        
        Args:
            query_vectors: Dict with keys 'original' and 'aspect_enhanced' (from DualEmbedder)
            k: Number of results to return
            alpha: Weight for aspect-enhanced index score (0=original only, 1=aspect only, 0.5=equal)
            where_filter: Optional ChromaDB metadata filter dict
                          e.g. {"doc_id": {"$eq": "doc_001"}}
        
        Returns:
            List of result dicts sorted by fused score, highest first.
            Each dict: { chunk_id, text, metadata, original_score, aspect_score, fused_score }
        
        SCORE FUSION FORMULA:
            fused = (1 - alpha) * orig_score + alpha * aspect_score
            Cosine distance [0,2] → similarity = 1 - distance/2  → range [0,1]
        """
        self._init_db()

        query_kwargs = {"n_results": k * 2}  # over-fetch then re-rank
        if where_filter:
            query_kwargs["where"] = where_filter

        # Query original index
        orig_results = self._orig_col.query(
            query_embeddings=[query_vectors["original"]],
            include=["documents", "metadatas", "distances"],
            **query_kwargs,
        )

        # Query aspect-enhanced index
        aspect_results = self._aspect_col.query(
            query_embeddings=[query_vectors["aspect_enhanced"]],
            include=["documents", "metadatas", "distances"],
            **query_kwargs,
        )

        # Build score maps: chunk_id → score
        orig_scores = {}
        for cid, dist in zip(
            orig_results["ids"][0], orig_results["distances"][0]
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity [0, 1]
            orig_scores[cid] = 1.0 - dist / 2.0

        aspect_scores = {}
        for cid, dist in zip(
            aspect_results["ids"][0], aspect_results["distances"][0]
        ):
            aspect_scores[cid] = 1.0 - dist / 2.0

        # Collect all unique candidates
        all_ids = set(orig_scores.keys()) | set(aspect_scores.keys())

        # Build result objects
        # We need documents and metadata — pull from whichever result set has them
        id_to_doc = {}
        id_to_meta = {}
        for cid, doc, meta in zip(
            orig_results["ids"][0],
            orig_results["documents"][0],
            orig_results["metadatas"][0],
        ):
            id_to_doc[cid] = doc
            id_to_meta[cid] = meta
        for cid, doc, meta in zip(
            aspect_results["ids"][0],
            aspect_results["documents"][0],
            aspect_results["metadatas"][0],
        ):
            if cid not in id_to_doc:
                id_to_doc[cid] = doc
                id_to_meta[cid] = meta

        # Fuse scores
        results = []
        for cid in all_ids:
            o_score = orig_scores.get(cid, 0.0)
            a_score = aspect_scores.get(cid, 0.0)
            fused = (1 - alpha) * o_score + alpha * a_score

            results.append({
                "chunk_id": cid,
                "text": id_to_doc.get(cid, ""),
                "metadata": id_to_meta.get(cid, {}),
                "original_score": round(o_score, 4),
                "aspect_score": round(a_score, 4),
                "fused_score": round(fused, 4),
                # Deserialize JSON-stored fields
                "key_topics": json.loads(id_to_meta.get(cid, {}).get("key_topics", "[]")),
                "contrastive_scope": json.loads(id_to_meta.get(cid, {}).get("contrastive_scope", "[]")),
            })

        # Sort by fused score, return top-k
        results.sort(key=lambda x: x["fused_score"], reverse=True)
        return results[:k]

    def apply_contrastive_penalty(
        self, results: list[dict], query: str, penalty: float = 0.15
    ) -> list[dict]:
        """
        Apply a score penalty to chunks whose contrastive_scope overlaps with the query.
        
        If a chunk says "does not cover: X" and the query asks about X,
        we slightly penalize that chunk — it's less likely to answer the question.
        
        Args:
            results: List of result dicts from query()
            query: The user's query string
            penalty: Score reduction for scope-mismatched chunks (default 0.15)
        
        Returns: Re-sorted results with penalties applied
        """
        query_lower = query.lower()

        for result in results:
            scope = result.get("contrastive_scope", [])
            if not scope:
                continue

            for item in scope:
                # Simple keyword overlap check
                item_words = set(item.lower().split())
                query_words = set(query_lower.split())
                overlap = item_words & query_words
                if len(overlap) >= 2:  # at least 2 words match
                    result["fused_score"] = max(0.0, result["fused_score"] - penalty)
                    result["contrastive_penalty_applied"] = True
                    break

        results.sort(key=lambda x: x["fused_score"], reverse=True)
        return results

    def count(self) -> dict:
        """Return number of chunks in each index."""
        self._init_db()
        return {
            "original_index": self._orig_col.count(),
            "aspect_enhanced_index": self._aspect_col.count(),
        }

    def reset(self):
        """Delete all data from both collections. Use carefully."""
        self._init_db()
        orig_name = f"{self.collection_prefix}_original"
        aspect_name = f"{self.collection_prefix}_aspect_enhanced"
        try:
            self._client.delete_collection(orig_name)
            self._client.delete_collection(aspect_name)
        except Exception:
            pass
        self._orig_col = None
        self._aspect_col = None
        self._init_db()
        print("  [VectorStore] Reset complete.")


if __name__ == "__main__":
    import random

    store = HybridVectorStore(persist_dir="./test_db")

    # Create fake chunks
    fake_chunks = []
    for i in range(3):
        dim = 384  # all-MiniLM-L6-v2 dimension
        fake_chunks.append({
            "chunk_id": f"chunk_{i:03d}",
            "doc_id": "test_doc",
            "text": f"This is test chunk {i} about machine learning and transformers.",
            "chunk_index": i,
            "original_embedding": [random.random() for _ in range(dim)],
            "aspect_enhanced_embedding": [random.random() for _ in range(dim)],
            "aspects": [],
            "summary": f"Summary of chunk {i}",
            "key_topics": ["machine learning", "transformers"],
            "contrastive_scope": ["image generation"],
            "aspect_text": "Key topics: machine learning, transformers.",
            "metadata": {"source": "test"},
        })

    store.add_chunks(fake_chunks)
    print(f"\nStore counts: {store.count()}")

    # Query
    query_vecs = {
        "original": [random.random() for _ in range(384)],
        "aspect_enhanced": [random.random() for _ in range(384)],
    }
    results = store.query(query_vecs, k=2)
    for r in results:
        print(f"  {r['chunk_id']}: fused={r['fused_score']:.4f}")

    store.reset()