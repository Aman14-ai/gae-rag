"""
GAE-RAG: Aspect Verifier (LLM-based)
====================================
Verifies that generated aspects are entailed by the source chunk using Groq LLM.
More accurate than NLI or embedding similarity for implied facts.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


VERIFICATION_SYSTEM_PROMPT = """You are an entailment judge for a RAG system.
Given a SOURCE CHUNK and a CLAIM, determine if the claim is logically entailed by the source.
Be generous with paraphrases and common-sense implications, but reject hallucinations.

Output ONLY a JSON object with two fields:
{
  "entailment": true/false,
  "confidence": 0.0-1.0
}

Rules:
- entailment = true if the source clearly implies or states the claim.
- entailment = false if the source contradicts the claim or says nothing about it.
- confidence = your certainty (0.8-1.0 for clear cases, 0.5-0.7 for vague implications).

Examples:

SOURCE: BERT is a transformer model trained with masked language modeling.
CLAIM: BERT uses self-attention.
OUTPUT: {"entailment": true, "confidence": 0.9}

SOURCE: BERT was released by Google in 2018.
CLAIM: BERT was released in 2019.
OUTPUT: {"entailment": false, "confidence": 1.0}

SOURCE: The model has 110M parameters.
CLAIM: The model is large.
OUTPUT: {"entailment": true, "confidence": 0.7}
"""


@dataclass
class VerificationResult:
    text: str
    entailment_score: float  # confidence from LLM
    label: str               # "ENTAILMENT" or "NEUTRAL"
    passed: bool


class   AspectVerifier:
    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.1-8b-instant",
        tau: float = 0.5,
        temperature: float = 0.0,
    ):
        """
        Args:
            groq_api_key: Your Groq API key
            model: Groq model (llama-3.1-8b-instant is fast and free)
            tau: Confidence threshold (0.5 = accept if confidence >= 0.5)
            temperature: 0.0 for deterministic output
        """
        self.llm = ChatGroq(api_key=groq_api_key, model=model, temperature=temperature)
        self.tau = tau

    def _aspect_to_text(self, aspect: dict) -> str:
        entity = aspect.get("entity", "")
        attribute = aspect.get("attribute", "")
        value = aspect.get("value", "")
        if entity and attribute and value:
            return f"{entity} {attribute} is {value}."
        elif entity and value:
            return f"{entity}: {value}."
        return value

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        content = content.strip()
        # Remove markdown code fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to find first { and last }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
            raise

    def verify_single(self, source_chunk: str, text_to_verify: str) -> VerificationResult:
        """
        Verify one claim against the source chunk using LLM.
        """
        prompt = f"""SOURCE CHUNK:
{source_chunk}

CLAIM:
{text_to_verify}

Determine if the claim is entailed by the source chunk. Output JSON."""
        
        messages = [
            SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self.llm.invoke(messages)
            parsed = self._extract_json(response.content)
            entailment = parsed.get("entailment", False)
            confidence = float(parsed.get("confidence", 0.0))
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            if not entailment:
                confidence = 0.0  # if not entailed, score 0 regardless of confidence
            passed = entailment and confidence >= self.tau
            label = "ENTAILMENT" if passed else "NEUTRAL"
            
            return VerificationResult(
                text=text_to_verify,
                entailment_score=round(confidence, 4),
                label=label,
                passed=passed,
            )
        except Exception as e:
            print(f"  [Verifier] Error: {e}")
            return VerificationResult(
                text=text_to_verify,
                entailment_score=0.0,
                label="ERROR",
                passed=False,
            )

    def verify_aspect_dict(self, source_chunk: str, aspect: dict) -> Tuple[dict, VerificationResult]:
        aspect_text = self._aspect_to_text(aspect)
        result = self.verify_single(source_chunk, aspect_text)
        aspect["verification_score"] = result.entailment_score
        aspect["verification_label"] = result.label
        return aspect, result

    def filter_aspects(
        self, source_chunk: str, raw_aspects: List[dict], verbose: bool = False
    ) -> Tuple[List[dict], dict]:
        verified = []
        rejected = []
        scores = []
        for aspect in raw_aspects:
            aspect_w_score, res = self.verify_aspect_dict(source_chunk, aspect)
            scores.append(res.entailment_score)
            if res.passed:
                verified.append(aspect_w_score)
            else:
                rejected.append(aspect_w_score)
                if verbose:
                    print(f"    REJECTED (score={res.entailment_score:.3f}): {res.text[:60]}...")
        stats = {
            "total": len(raw_aspects),
            "passed": len(verified),
            "rejected": len(rejected),
            "avg_score": round(sum(scores)/len(scores),4) if scores else 0.0,
            "tau_used": self.tau,
        }
        return verified, stats

    def filter_related_context(self, source_chunk: str, related_context: List[dict]) -> List[dict]:
        verified = []
        for ctx in related_context:
            text = ctx.get("text", "")
            if not text:
                continue
            result = self.verify_single(source_chunk, text)
            if result.passed:
                ctx["verification_score"] = result.entailment_score
                verified.append(ctx)
        return verified

    def verify_full_aspect_data(self, aspect_data: dict, verbose: bool = True) -> dict:
        source = aspect_data.get("raw_chunk", "")
        chunk_id = aspect_data.get("chunk_id", "?")
        if verbose:
            print(f"\n  [Verifier] Chunk: {chunk_id}")
        
        aspects = aspect_data.get("aspects", [])
        if aspects:
            verified_aspects, stats = self.filter_aspects(source, aspects, verbose=verbose)
            aspect_data["aspects"] = verified_aspects
            aspect_data["verification_stats"] = stats
            if verbose:
                print(f"  [Verifier] Aspects: {stats['passed']}/{stats['total']} passed (avg score: {stats['avg_score']:.3f}, tau={self.tau})")
        else:
            aspect_data["verification_stats"] = {"total":0,"passed":0,"rejected":0,"avg_score":0.0,"tau_used":self.tau}
        
        related = aspect_data.get("related_context", [])
        if related:
            verified_ctx = self.filter_related_context(source, related)
            aspect_data["related_context"] = verified_ctx
            if verbose:
                print(f"  [Verifier] Related context: {len(verified_ctx)}/{len(related)} passed")
        return aspect_data


if __name__ == "__main__":
    import os
    api_key = os.environ.get("GROQ_API_KEY", "your-groq-key-here")
    verifier = AspectVerifier(groq_api_key=api_key, tau=0.5)
    
    source = """
    BERT is a transformer model trained with masked language modeling.
    It uses bidirectional attention to understand context from both sides.
    BERT was released by Google in 2018 and has 110M parameters in its base form.
    """
    
    r1 = verifier.verify_single(source, "BERT was released by Google in 2018.")
    print(f"Test 1 (should PASS): {r1.label} score={r1.entailment_score} passed={r1.passed}")
    
    r2 = verifier.verify_single(source, "BERT can generate images from text.")
    print(f"Test 2 (should FAIL): {r2.label} score={r2.entailment_score} passed={r2.passed}")
    
    r3 = verifier.verify_single(source, "BERT uses self-attention to process text.")
    print(f"Test 3 (should PASS): {r3.label} score={r3.entailment_score} passed={r3.passed}")