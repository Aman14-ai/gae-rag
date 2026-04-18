import re
import hashlib
from typing import Optional, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Splits raw text into overlapping chunks for indexing using LangChain's RecursiveCharacterTextSplitter.
    
    Strategy (handled by LangChain):
      1. Recursively split on separators: ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
      2. Maintains overlap by taking whole words/sentences from previous chunk
      3. Falls back to character split only as last resort
    
    Args:
        max_chunk_chars: Max characters per chunk (default 800 ~ 150-200 words)
        overlap_chars:   Characters of overlap between adjacent chunks (default 150)
        min_chunk_chars: Discard chunks shorter than this (default 100)
        separators:      Optional custom separators (default: paragraph → sentence → word)
    """

    def __init__(
        self,
        max_chunk_chars: int = 800,
        overlap_chars: int = 150,
        min_chunk_chars: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        
        # LangChain splitter – handles overlap & recursive splitting
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=overlap_chars,
            separators=self.separators,
            length_function=len,          # character count; can be swapped for token count
            keep_separator=True,          # keeps punctuation at chunk boundaries
        )

    def _make_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
        return f"{doc_id}_chunk{index:03d}_{text_hash}"

    def chunk_document(self, text: str, doc_id: str = "doc") -> List[Dict]:
        """
        Chunk a single document string.
        
        Returns list of dicts:
          { 'chunk_id', 'doc_id', 'text', 'chunk_index' }
        """
        # LangChain returns a list of chunk strings (already overlapped)
        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < self.min_chunk_chars:
                continue
            chunks.append({
                "chunk_id": self._make_chunk_id(doc_id, idx, chunk_text),
                "doc_id": doc_id,
                "text": chunk_text.strip(),
                "chunk_index": idx,
            })
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        Each doc dict must have: 'text', optionally 'doc_id' and 'metadata'.
        
        Returns flat list of all chunks with doc metadata attached.
        """
        all_chunks = []
        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(all_chunks)}")
            doc_chunks = self.chunk_document(doc["text"], doc_id=doc_id)
            # Attach any doc-level metadata to each chunk
            for chunk in doc_chunks:
                chunk["metadata"] = doc.get("metadata", {})
            all_chunks.extend(doc_chunks)
        print(f"  [Chunker] {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks


# ─────────────────────────────────────────────
# DUAL EMBEDDER (unchanged, works perfectly)
# ─────────────────────────────────────────────

class DualEmbedder:
    """
    Produces two embeddings per chunk:
      1. original_embedding       — from raw chunk text alone
      2. aspect_enhanced_embedding — from chunk + aspects + summary
    
    Uses sentence-transformers (runs locally, no API key needed).
    Model: all-MiniLM-L6-v2 (~22MB, very fast, good quality)
    
    WHY DUAL EMBEDDINGS?
    If a user asks "What are the side effects of Drug X?" but the chunk only
    says "Drug X causes nausea, fatigue, and dizziness" — the original embedding
    might not match "side effects" well. The aspect-enhanced embedding includes
    the aspect "Drug X side_effects: nausea, fatigue, dizziness" which aligns
    much better with the query.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"  [Embedder] Loading model: {self.model_name}")
            print("  [Embedder] First run downloads ~22MB — please wait...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print("  [Embedder] Model loaded.")

    def _build_aspect_text(self, aspect_data: dict) -> str:
        """
        Build the aspect-enhanced text string from verified aspect data.
        Format: [summary] + [aspects as sentences] + [related context]
        """
        parts = []

        # Summary
        summary = aspect_data.get("summary", "")
        if summary:
            parts.append(summary)

        # Key topics
        topics = aspect_data.get("key_topics", [])
        if topics:
            parts.append("Topics: " + ", ".join(topics))

        # Aspects → natural language
        for asp in aspect_data.get("aspects", []):
            entity = asp.get("entity", "")
            attr = asp.get("attribute", "")
            value = asp.get("value", "")
            if entity and value:
                parts.append(f"{entity} {attr}: {value}.")

        # Related context
        for ctx in aspect_data.get("related_context", []):
            text = ctx.get("text", "")
            if text:
                parts.append(text)

        return " ".join(parts)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns list of floats."""
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_chunk_pair(self, chunk_text: str, aspect_data: dict) -> dict:
        """
        Embed a single chunk and its aspects.
        
        Returns:
            {
              'original_embedding': [...],
              'aspect_enhanced_embedding': [...],
              'aspect_text': "..."   (the constructed aspect string)
            }
        """
        self._load_model()

        aspect_text = self._build_aspect_text(aspect_data)
        combined_text = chunk_text + " " + aspect_text

        originals = self._model.encode(
            [chunk_text, combined_text], normalize_embeddings=True
        )

        return {
            "original_embedding": originals[0].tolist(),
            "aspect_enhanced_embedding": originals[1].tolist(),
            "aspect_text": aspect_text,
        }

    def embed_query(self, query: str) -> dict:
        """
        Embed a query for both indexes.
        Returns: { 'original': [...], 'aspect_enhanced': [...] }
        """
        self._load_model()
        vectors = self._model.encode(
            [query, query], normalize_embeddings=True
        )
        return {
            "original": vectors[0].tolist(),
            "aspect_enhanced": vectors[1].tolist(),
        }


# ─────────────────────────────────────────────
# EXAMPLE USAGE (same as before)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test chunker
    chunker = SemanticChunker(max_chunk_chars=500, overlap_chars=100)
    test_doc = {
        "doc_id": "ml_intro",
        "text": """
        Machine learning is a subset of artificial intelligence that enables systems to learn 
        from data without being explicitly programmed. It focuses on developing algorithms that 
        can access data and use it to learn for themselves.

        Supervised learning is one of the most common types of machine learning. In supervised 
        learning, the model is trained on labeled data — meaning each training example has an 
        input and a known correct output. The model learns to map inputs to outputs.

        Unsupervised learning deals with unlabeled data. The model tries to find patterns and 
        structure in the data on its own. Clustering and dimensionality reduction are common 
        unsupervised learning tasks.
        """,
        "metadata": {"source": "textbook", "chapter": 1}
    }

    chunks = chunker.chunk_documents([test_doc])
    for c in chunks:
        print(f"Chunk {c['chunk_id']}: {len(c['text'])} chars")
        print(f"  Preview: {c['text'][:100]}...\n")

    # Test embedder
    embedder = DualEmbedder()
    sample_aspect_data = {
        "summary": "Machine learning enables systems to learn from data.",
        "key_topics": ["machine learning", "supervised learning", "algorithms"],
        "aspects": [
            {"entity": "supervised learning", "attribute": "data_type", "value": "labeled data", "confidence": 0.98}
        ],
        "related_context": [
            {"text": "ML algorithms improve their performance as they process more data.", "type": "implication"}
        ],
    }

    if chunks:
        result = embedder.embed_chunk_pair(chunks[0]["text"], sample_aspect_data)
        print(f"Original embedding dim: {len(result['original_embedding'])}")
        print(f"Aspect-enhanced dim:    {len(result['aspect_enhanced_embedding'])}")
        print(f"Aspect text: {result['aspect_text'][:150]}...")