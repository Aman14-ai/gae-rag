"""
GAE-RAG: Adaptive Retrieval Module
=====================================
Three-stage retrieval with confidence-based fallback:

  Stage 1: Initial retrieval from dual index
  Stage 2: Confidence check → if low, rewrite query and retry
  Stage 3: Aspect-aware expansion — enrich query with aspect terms from top results
            and do a final pass to catch missed chunks

This module implements the most novel part of GAE-RAG's retrieval.
"""

import re
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


QUERY_REWRITE_PROMPT = """You are a query rewriting assistant for a RAG system.
The original query retrieved low-confidence results, meaning the vector database
could not find closely matching chunks.

Your task: Rewrite the query in 2-3 alternative phrasings that might match the 
document content better. Think about:
- Synonyms and alternative terms
- More specific or more general versions
- Different word orders
- Aspect-focused rewrites (e.g., "What are the effects of X" → "X side effects")

Return ONLY a JSON array of 2-3 alternative queries, no explanation:
["rewrite 1", "rewrite 2", "rewrite 3"]"""


ASPECT_EXPANSION_PROMPT = """You are a query expansion assistant for a RAG system.
Given:
- The original user query
- A list of key topics/aspects from the top retrieved chunks

Generate an expanded query that incorporates the most relevant aspects from the retrieved
context to find additional related chunks.

Return ONLY the expanded query string, nothing else."""


class AdaptiveRetriever:
    """
    Adaptive retrieval with three stages:
      1. Initial dual-index retrieval
      2. Confidence check + query rewrite (if needed)
      3. Aspect-aware expansion pass
    
    Args:
        vector_store: HybridVectorStore instance
        embedder: DualEmbedder instance
        groq_api_key: For query rewriting/expansion LLM calls
        confidence_threshold: Min fused score to consider retrieval "confident"
        model: Groq model for rewriting (small model is fine)
        initial_k: Number of results in stage 1
        expansion_k: Extra results in stage 3
    """

    def __init__(
        self,
        vector_store,
        embedder,
        groq_api_key: str,
        confidence_threshold: float = 0.45,
        model: str = "llama-3.1-8b-instant",
        initial_k: int = 5,
        expansion_k: int = 3,
    ):
        self.store = vector_store
        self.embedder = embedder
        self.confidence_threshold = confidence_threshold
        self.initial_k = initial_k
        self.expansion_k = expansion_k

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=0.3,
            max_tokens=512,
        )

    def _check_confidence(self, results: list[dict]) -> float:
        """
        Compute retrieval confidence: average of top-3 fused scores.
        Range: [0, 1] — above threshold means we're confident in results.
        """
        if not results:
            return 0.0
        top_scores = [r["fused_score"] for r in results[:3]]
        return sum(top_scores) / len(top_scores)

    def _rewrite_query(self, original_query: str) -> list[str]:
        """Use LLM to generate alternative query phrasings."""
        import json, re
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=f"Original query: {original_query}"),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            rewrites = json.loads(content)
            if isinstance(rewrites, list):
                return [str(r) for r in rewrites]
        except Exception as e:
            print(f"  [Retriever] Query rewrite failed: {e}")
        return [original_query]

    def _expand_query_with_aspects(
        self, original_query: str, top_results: list[dict]
    ) -> str:
        """
        Build an aspect-enriched query from the top results' key topics.
        This helps surface chunks that are thematically related but
        didn't match the original query phrasing.
        """
        # Collect aspects from top results
        all_topics = []
        for result in top_results[:3]:
            topics = result.get("key_topics", [])
            all_topics.extend(topics)

        if not all_topics:
            return original_query

        # Deduplicate
        unique_topics = list(dict.fromkeys(all_topics))[:8]
        topic_str = ", ".join(unique_topics)

        messages = [
            SystemMessage(content=ASPECT_EXPANSION_PROMPT),
            HumanMessage(
                content=(
                    f"Original query: {original_query}\n"
                    f"Key aspects from retrieved chunks: {topic_str}\n\n"
                    "Generate an expanded query:"
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            expanded = response.content.strip().strip('"').strip("'")
            return expanded if expanded else original_query
        except Exception:
            return original_query

    def _merge_results(
        self,
        primary: list[dict],
        secondary: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Merge two result lists, deduplicating by chunk_id.
        Primary results keep their score; secondary results are gently
        down-weighted (×0.85) to prefer primary retrieval.
        """
        seen_ids = set()
        merged = []

        for r in primary:
            if r["chunk_id"] not in seen_ids:
                merged.append(r)
                seen_ids.add(r["chunk_id"])

        for r in secondary:
            if r["chunk_id"] not in seen_ids:
                r = dict(r)
                r["fused_score"] *= 0.85  # slight down-weight
                r["from_expansion"] = True
                merged.append(r)
                seen_ids.add(r["chunk_id"])

        merged.sort(key=lambda x: x["fused_score"], reverse=True)
        return merged[:top_k]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Full three-stage adaptive retrieval.
        
        Args:
            query: User's natural language query
            k: Final number of chunks to return
            alpha: Weight for aspect-enhanced index (0.0-1.0)
            verbose: Print retrieval stages
        
        Returns dict:
          {
            'results': list[dict],          # final ranked chunks
            'query_used': str,              # original or rewritten
            'expanded_query': str,          # aspect-expanded query (stage 3)
            'confidence': float,            # initial retrieval confidence
            'rewrites_used': bool,          # whether rewriting was triggered
            'expansion_used': bool,         # whether expansion ran
            'stage_confidences': [float],   # confidence after each stage
          }
        """
        retrieval_info = {
            "query_used": query,
            "expanded_query": query,
            "confidence": 0.0,
            "rewrites_used": False,
            "expansion_used": False,
            "stage_confidences": [],
        }

        # ─── STAGE 1: Initial retrieval ───
        if verbose:
            print(f"\n  [Retriever] Stage 1: Initial retrieval for: '{query[:80]}'")

        query_vecs = self.embedder.embed_query(query)
        results = self.store.query(query_vecs, k=self.initial_k, alpha=alpha)

        # Apply contrastive penalty
        results = self.store.apply_contrastive_penalty(results, query)

        confidence = self._check_confidence(results)
        retrieval_info["confidence"] = confidence
        retrieval_info["stage_confidences"].append(confidence)

        if verbose:
            print(f"  [Retriever] Stage 1 confidence: {confidence:.3f} (threshold: {self.confidence_threshold})")

        # ─── STAGE 2: Query rewrite if low confidence ───
        if confidence < self.confidence_threshold:
            if verbose:
                print("  [Retriever] Stage 2: Low confidence — rewriting query...")

            rewrites = self._rewrite_query(query)
            retrieval_info["rewrites_used"] = True
            best_results = results
            best_confidence = confidence

            for rewrite in rewrites:
                if verbose:
                    print(f"  [Retriever]   Trying: '{rewrite[:70]}'")

                rw_vecs = self.embedder.embed_query(rewrite)
                rw_results = self.store.query(rw_vecs, k=self.initial_k, alpha=alpha)
                rw_results = self.store.apply_contrastive_penalty(rw_results, rewrite)
                rw_confidence = self._check_confidence(rw_results)

                if rw_confidence > best_confidence:
                    best_confidence = rw_confidence
                    best_results = rw_results
                    retrieval_info["query_used"] = rewrite

            results = best_results
            retrieval_info["stage_confidences"].append(best_confidence)

            if verbose:
                print(f"  [Retriever] Stage 2 best confidence: {best_confidence:.3f}")

        # ─── STAGE 3: Aspect-aware expansion ───
        if verbose:
            print("  [Retriever] Stage 3: Aspect-aware expansion...")

        expanded_query = self._expand_query_with_aspects(
            retrieval_info["query_used"], results
        )

        if expanded_query != retrieval_info["query_used"]:
            retrieval_info["expanded_query"] = expanded_query
            retrieval_info["expansion_used"] = True

            if verbose:
                print(f"  [Retriever]   Expanded: '{expanded_query[:80]}'")

            exp_vecs = self.embedder.embed_query(expanded_query)
            exp_results = self.store.query(exp_vecs, k=self.expansion_k, alpha=alpha)
            exp_results = self.store.apply_contrastive_penalty(exp_results, expanded_query)

            results = self._merge_results(results, exp_results, top_k=k)
            final_confidence = self._check_confidence(results)
            retrieval_info["stage_confidences"].append(final_confidence)
        else:
            results = results[:k]

        retrieval_info["results"] = results

        if verbose:
            print(f"  [Retriever] Final: {len(results)} chunks retrieved")
            for i, r in enumerate(results):
                prefix = "(expanded)" if r.get("from_expansion") else ""
                print(
                    f"    {i+1}. [{r['chunk_id']}] fused={r['fused_score']:.3f} "
                    f"orig={r['original_score']:.3f} asp={r['aspect_score']:.3f} {prefix}"
                )

        return retrieval_info