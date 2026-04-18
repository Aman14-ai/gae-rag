"""
GAE-RAG: Reranker + Answer Generator + Grounding Check
=========================================================
Three final pipeline stages:

1. Reranker          — cross-encoder reranking of retrieved chunks
2. AnswerGenerator   — LLM answer with aspect-aware context formatting
3. GroundingChecker  — LLM‑based hallucination detection + retry loop
"""

import json
import re
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-encoder reranker using a local model.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - ~67MB download on first run
      - Specifically trained for passage reranking (MS MARCO)
      - Takes (query, passage) pair → relevance score
    
    WHY RERANK after dual-index retrieval?
    The dual embeddings are approximate (bi-encoder) — they find
    semantically close chunks. The cross-encoder reads both query and
    passage together for a more precise relevance judgment.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"  [Reranker] Loading: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("  [Reranker] Model loaded.")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank retrieved chunks by query-passage relevance.
        
        Args:
            query: The user query
            results: List of result dicts from AdaptiveRetriever
            top_k: How many to return after reranking (default: all)
        
        Returns:
            Reranked list with 'reranker_score' added to each result.
        """
        if not results:
            return results

        self._load_model()

        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result["reranker_score"] = float(round(score, 4))

        reranked = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_k] if top_k else reranked


# ─────────────────────────────────────────────
# ANSWER GENERATOR
# ─────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are a precise, helpful assistant. Answer the user's question 
using ONLY the information provided in the context chunks below.

RULES:
1. Base your answer entirely on the provided context — do not use outside knowledge.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite which chunk(s) support each key claim by referencing [Chunk N].
4. Be concise but complete. Do not pad with unnecessary text.
5. If aspects or key topics are listed for a chunk, use them to understand the chunk's focus.

Context format:
--- Chunk N [chunk_id] ---
Text: ...
Summary: ...
Key topics: ...
---"""

class AnswerGenerator:
    """
    Generates a grounded answer from retrieved chunks.
    
    Args:
        groq_api_key: Groq API key
        model: Model to use for generation (larger = better quality)
        temperature: 0.1 for factual answers, 0.3 for more explanatory
        max_tokens: Max answer length
    """

    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.1-70b-versatile",   # larger model for better answers
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a structured context string."""
        parts = []
        for i, r in enumerate(results, 1):
            chunk_id = r.get("chunk_id", f"chunk_{i}")
            text = r.get("text", "")
            summary = r.get("metadata", {}).get("summary", "")
            topics = r.get("key_topics", [])

            part = f"--- Chunk {i} [{chunk_id}] ---\n"
            part += f"Text: {text}\n"
            if summary:
                part += f"Summary: {summary}\n"
            if topics:
                part += f"Key topics: {', '.join(topics)}\n"
            part += "---"
            parts.append(part)

        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        results: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Generate an answer from the query + retrieved chunks.
        
        Returns:
          {
            'answer': str,
            'context_used': str,  # formatted context sent to LLM
            'num_chunks': int,
            'error': str or None,
          }
        """
        context = self._format_context(results)

        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}"
            ),
        ]

        output = {
            "answer": "",
            "context_used": context,
            "num_chunks": len(results),
            "error": None,
        }

        try:
            response = self.llm.invoke(messages)
            output["answer"] = response.content.strip()
            if verbose:
                print(f"\n  [Generator] Answer generated ({len(output['answer'])} chars)")
        except Exception as e:
            output["error"] = str(e)
            output["answer"] = "Error generating answer."
            if verbose:
                print(f"  [Generator] Error: {e}")

        return output


# ─────────────────────────────────────────────
# GROUNDING CHECKER (LLM‑based, no NLI)
# ─────────────────────────────────────────────

CLAIM_EXTRACTOR_PROMPT = """Extract all factual claims from the following answer as a JSON list of strings.
Answer: {answer}
Output only a JSON list, e.g., ["claim 1", "claim 2"]"""

ENTAILMENT_PROMPT = """Does the following CLAIM logically follow from the provided CONTEXT? Answer only "YES" or "NO".

CONTEXT:
{context}

CLAIM:
{claim}

Output: YES/NO"""

class GroundingChecker:
    """
    Checks whether an LLM-generated answer is grounded in the retrieved context.
    
    Process:
      1. Extract individual claims from the answer (using LLM)
      2. For each claim, ask the LLM whether it is entailed by the concatenated context
      3. Compute attribution rate = grounded_claims / total_claims
      4. If attribution rate < threshold → flag as potentially hallucinated
    
    Args:
        groq_api_key: For claim extraction and entailment
        grounding_threshold: Min attribution rate to accept answer (default 0.7)
        model: LLM model for both extraction and entailment (small model works fine)
    """

    def __init__(
        self,
        groq_api_key: str,
        grounding_threshold: float = 0.7,
        model: str = "llama-3.1-8b-instant",
    ):
        self.grounding_threshold = grounding_threshold
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=0.0,
            max_tokens=512,
        )

    def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from the answer using LLM."""
        prompt = CLAIM_EXTRACTOR_PROMPT.format(answer=answer)
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            claims = json.loads(content)
            if isinstance(claims, list):
                return [str(c) for c in claims if c]
        except Exception as e:
            print(f"  [Grounding] Claim extraction error: {e}")
        return []

    def _check_claim(self, claim: str, context_chunks: list[dict]) -> dict:
        """
        Check if a claim is entailed by the retrieved chunks using LLM.
        
        Returns:
          { 'claim': str, 'grounded': bool, 'best_score': float, 'best_chunk': str }
        """
        # Concatenate all chunks into a single context
        context = "\n\n---\n\n".join([c.get("text", "") for c in context_chunks])
        prompt = ENTAILMENT_PROMPT.format(context=context, claim=claim)
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()
            grounded = answer.startswith("YES")
            return {
                "claim": claim,
                "grounded": grounded,
                "best_score": 1.0 if grounded else 0.0,
                "best_chunk": "",  # LLM doesn't tell which chunk; not needed
            }
        except Exception as e:
            print(f"  [Grounding] Entailment error: {e}")
            return {"claim": claim, "grounded": False, "best_score": 0.0, "best_chunk": ""}

    def check(
        self,
        answer: str,
        context_results: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Run full grounding check on an answer.
        
        Returns:
          {
            'grounded': bool,            # True if attribution_rate >= threshold
            'attribution_rate': float,   # fraction of claims that are grounded
            'total_claims': int,
            'grounded_claims': int,
            'ungrounded_claims': list[str],  # claims that failed grounding
            'claim_details': list[dict],
            'retry_needed': bool,
          }
        """
        result = {
            "grounded": True,
            "attribution_rate": 1.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "ungrounded_claims": [],
            "claim_details": [],
            "retry_needed": False,
        }

        if verbose:
            print("\n  [Grounding] Extracting claims from answer...")

        claims = self._extract_claims(answer)
        result["total_claims"] = len(claims)

        if not claims:
            if verbose:
                print("  [Grounding] No claims extracted — skipping verification.")
            return result

        if verbose:
            print(f"  [Grounding] Checking {len(claims)} claims against {len(context_results)} chunks...")

        grounded_count = 0
        for claim in claims:
            detail = self._check_claim(claim, context_results)
            result["claim_details"].append(detail)

            if detail["grounded"]:
                grounded_count += 1
                if verbose:
                    print(f"  [Grounding]   ✓ {claim[:70]}")
            else:
                result["ungrounded_claims"].append(claim)
                if verbose:
                    print(f"  [Grounding]   ✗ {claim[:70]}")

        result["grounded_claims"] = grounded_count
        attribution_rate = grounded_count / len(claims) if claims else 1.0
        result["attribution_rate"] = round(attribution_rate, 4)
        result["grounded"] = attribution_rate >= self.grounding_threshold
        result["retry_needed"] = not result["grounded"]

        if verbose:
            status = "PASSED" if result["grounded"] else "FAILED — RETRY NEEDED"
            print(
                f"\n  [Grounding] {status} | Attribution: "
                f"{grounded_count}/{len(claims)} = {attribution_rate:.1%} "
                f"(threshold: {self.grounding_threshold:.0%})"
            )

        return result