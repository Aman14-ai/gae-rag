"""
GAE-RAG: Full Pipeline Orchestrator
=====================================
Ties all modules together into a single class.

INDEXING PIPELINE:
  Documents → Chunker → AspectGenerator → AspectVerifier → DualEmbedder → VectorStore

QUERY PIPELINE:
  Query → AdaptiveRetriever → CrossEncoderReranker → AnswerGenerator → GroundingChecker
  (with retry loop if grounding fails)

Usage:
  from pipeline import GAERAGPipeline
  
  rag = GAERAGPipeline(groq_api_key="gsk_...")
  rag.index(documents)               # one-time indexing
  answer = rag.query("Your question here")
"""

import json
import os
from typing import Optional

from core.chunker_embedder import SemanticChunker, DualEmbedder
from core.aspect_generator import AspectGenerator
from core.aspect_verifier import AspectVerifier
from core.vector_store import HybridVectorStore
from core.retriever import AdaptiveRetriever
from core.generator_grounding import CrossEncoderReranker, AnswerGenerator, GroundingChecker


class GAERAGPipeline:
    """
    Full Grounded Aspect Expansion RAG pipeline.
    
    Args:
        groq_api_key:         Your Groq API key (get at console.groq.com — free tier)
        persist_dir:          Where to store ChromaDB data
        generation_model:     Larger model for answer quality (llama3-70b or mixtral)
        utility_model:        Smaller fast model for rewriting/extraction
        nli_tau:              NLI threshold for aspect verification (0.4-0.7)
        grounding_threshold:  Min attribution rate before retry (0.6-0.8)
        confidence_threshold: Min retrieval score before query rewrite (0.4-0.5)
        max_retries:          Max grounding-based answer retries
        use_reranker:         Whether to run cross-encoder reranking
    """

    def __init__(
        self,
        groq_api_key: str,
        persist_dir: str = "./gae_rag_db",
        generation_model: str = "llama-3.1-8b-instant",
        utility_model: str = "llama-3.1-8b-instant",
        nli_tau: float = 0.5,
        grounding_threshold: float = 0.7,
        confidence_threshold: float = 0.45,
        max_retries: int = 2,
        use_reranker: bool = True,
    ):
        self.groq_api_key = groq_api_key
        self.max_retries = max_retries
        self.use_reranker = use_reranker

        print("Initializing GAE-RAG pipeline components...")

        # Indexing components
        self.chunker = SemanticChunker(
            max_chunk_chars=800,
            overlap_chars=150,
            min_chunk_chars=100,
        )
        self.aspect_generator = AspectGenerator(
            groq_api_key=groq_api_key,
            model=utility_model,
        )
        self.aspect_verifier = AspectVerifier(tau=nli_tau,groq_api_key=groq_api_key)
        self.embedder = DualEmbedder()

        # Storage
        self.vector_store = HybridVectorStore(persist_dir=persist_dir)

        # Retrieval components
        self.retriever = AdaptiveRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            groq_api_key=groq_api_key,
            confidence_threshold=confidence_threshold,
            model=utility_model,
        )

        # Generation components
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.answer_generator = AnswerGenerator(
            groq_api_key=groq_api_key,
            model=generation_model,
        )
        self.grounding_checker = GroundingChecker(
            groq_api_key=groq_api_key,
            grounding_threshold=grounding_threshold,
            model=utility_model,
        )

        print("Pipeline ready.\n")

    # ─────────────────────────────────────────
    # INDEXING
    # ─────────────────────────────────────────

    def index(self, documents: list[dict], verbose: bool = True) -> dict:
        """
        Index a list of documents through the full indexing pipeline.
        
        Args:
            documents: List of dicts, each with:
                       - 'text'    (required): document content
                       - 'doc_id'  (optional): unique ID
                       - 'metadata' (optional): dict of extra fields
            verbose: Print progress
        
        Returns:
            Stats dict with counts and error info
        
        EXAMPLE:
            docs = [
                {"text": "...", "doc_id": "paper_001", "metadata": {"source": "arxiv"}},
                {"text": "...", "doc_id": "paper_002"},
            ]
            stats = rag.index(docs)
        """
        stats = {
            "documents": len(documents),
            "chunks_created": 0,
            "aspects_generated": 0,
            "aspects_verified": 0,
            "indexing_errors": 0,
        }

        print(f"\n{'='*60}")
        print(f" INDEXING {len(documents)} DOCUMENT(S)")
        print(f"{'='*60}")

        # Step 1: Chunk
        print("\n[Step 1/4] Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        stats["chunks_created"] = len(chunks)

        # Step 2: Generate aspects
        print("\n[Step 2/4] Generating aspects (LLM)...")
        aspect_data_list = self.aspect_generator.generate_batch(chunks)

        total_raw = sum(len(a.get("aspects", [])) for a in aspect_data_list)
        stats["aspects_generated"] = total_raw

        # Step 3: Verify aspects
        print("\n[Step 3/4] Verifying aspects (NLI)...")
        for i, aspect_data in enumerate(aspect_data_list):
            if aspect_data.get("generation_error"):
                stats["indexing_errors"] += 1
                if verbose:
                    print(f"  Chunk {i}: SKIP (gen error: {aspect_data['generation_error'][:60]})")
                continue
            aspect_data_list[i] = self.aspect_verifier.verify_full_aspect_data(
                aspect_data, verbose=verbose
            )

        total_verified = sum(len(a.get("aspects", [])) for a in aspect_data_list)
        stats["aspects_verified"] = total_verified

        # Step 4: Embed + index
        print("\n[Step 4/4] Embedding + indexing into ChromaDB...")
        indexed_chunks = []
        for chunk, aspect_data in zip(chunks, aspect_data_list):
            embedding_result = self.embedder.embed_chunk_pair(
                chunk["text"], aspect_data
            )
            merged = {**chunk, **aspect_data, **embedding_result}
            indexed_chunks.append(merged)

        self.vector_store.add_chunks(indexed_chunks)

        print(f"\n{'='*60}")
        print(f" INDEXING COMPLETE")
        print(f"  Documents:          {stats['documents']}")
        print(f"  Chunks:             {stats['chunks_created']}")
        print(f"  Aspects generated:  {stats['aspects_generated']}")
        print(f"  Aspects verified:   {stats['aspects_verified']}")
        print(f"  Errors:             {stats['indexing_errors']}")
        print(f"{'='*60}\n")

        return stats

    # ─────────────────────────────────────────
    # QUERYING
    # ─────────────────────────────────────────

    def query(
        self,
        question: str,
        k: int = 5,
        alpha: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Answer a question using the full GAE-RAG pipeline.
        
        Args:
            question: Natural language question
            k:        Number of chunks to retrieve
            alpha:    Aspect-index weight for score fusion (0=original, 1=aspect, 0.5=equal)
            verbose:  Print pipeline stages
        
        Returns dict:
          {
            'question': str,
            'answer': str,
            'retrieval': dict,         # from AdaptiveRetriever
            'grounding': dict,         # from GroundingChecker
            'reranked_chunks': list,   # final chunks used
            'retries': int,            # how many retry loops ran
            'final_grounded': bool,
          }
        """
        output = {
            "question": question,
            "answer": "",
            "retrieval": {},
            "grounding": {},
            "reranked_chunks": [],
            "retries": 0,
            "final_grounded": False,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f" QUERY: {question}")
            print(f"{'='*60}")

        # ── Retrieval ──
        retrieval_info = self.retriever.retrieve(
            query=question, k=k, alpha=alpha, verbose=verbose
        )
        output["retrieval"] = retrieval_info
        results = retrieval_info["results"]

        # ── Reranking ──
        if self.use_reranker and results:
            if verbose:
                print(f"\n  [Reranker] Reranking {len(results)} chunks...")
            results = self.reranker.rerank(question, results, top_k=k)
            if verbose:
                print(f"  [Reranker] Done. Top score: {results[0]['reranker_score']:.4f}")

        output["reranked_chunks"] = results

        # ── Generation + Grounding Retry Loop ──
        current_question = question
        for attempt in range(self.max_retries + 1):
            if verbose and attempt > 0:
                print(f"\n  [Pipeline] RETRY {attempt}/{self.max_retries}...")

            gen_output = self.answer_generator.generate(
                current_question, results, verbose=verbose
            )
            answer = gen_output["answer"]

            grounding = self.grounding_checker.check(
                answer, results, verbose=verbose
            )

            if grounding["grounded"] or attempt == self.max_retries:
                output["answer"] = answer
                output["grounding"] = grounding
                output["retries"] = attempt
                output["final_grounded"] = grounding["grounded"]
                break

            # Build a focused retry question from ungrounded claims
            ungrounded = grounding.get("ungrounded_claims", [])
            if ungrounded:
                ungrounded_str = "; ".join(ungrounded[:3])
                current_question = (
                    f"{question}\n\n"
                    f"Note: Specifically verify these claims against the context: {ungrounded_str}"
                )

        if verbose:
            status = "✓ GROUNDED" if output["final_grounded"] else "⚠ UNGROUNDED (max retries)"
            print(f"\n{'='*60}")
            print(f" STATUS: {status} | Retries: {output['retries']}")
            print(f"{'='*60}")
            print(f"\nANSWER:\n{output['answer']}")
            print(f"{'='*60}\n")

        return output

    def get_store_stats(self) -> dict:
        """Return current vector store statistics."""
        return self.vector_store.count()

    def reset_index(self):
        """Clear all indexed data. Useful for re-indexing."""
        self.vector_store.reset()
        print("Index cleared.")