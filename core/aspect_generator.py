"""
GAE-RAG: Aspect Generator
=========================
Takes a text chunk and generates structured aspects using a constrained LLM call.
Aspects = entity-attribute pairs + related context + contrastive scope tags.
"""

import json
import re
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


ASPECT_SYSTEM_PROMPT = """You are an expert knowledge extractor for a RAG (Retrieval-Augmented Generation) system.

Given a text chunk, extract structured aspects in valid JSON format.

DEFINITIONS:
- "aspects": List of entity-attribute pairs explicitly supported by the text. 
  Each aspect has: "entity", "attribute", "value", "confidence" (0.0-1.0)
- "related_context": List of closely related concepts/facts that expand on the text's meaning.
  Each has: "text" (short sentence), "type" (one of: "elaboration", "implication", "background")
- "contrastive_scope": Things this chunk explicitly does NOT cover. 
  Used for negative scoring at retrieval time — do NOT invent, only state clear omissions.
- "summary": 1-sentence summary of the chunk.
- "key_topics": List of 3-6 main topics/keywords.

RULES:
1. Only extract what is clearly supported by the text — never hallucinate.
2. Confidence reflects how directly the text supports the aspect (1.0 = stated verbatim).
3. related_context must be logically entailed or strongly implied — not speculation.
4. contrastive_scope entries must be things a user might reasonably expect this chunk to cover but it doesn't.
5. Return ONLY valid JSON, no explanation, no markdown, no code fences.

OUTPUT FORMAT (strict):
{
  "summary": "...",
  "key_topics": ["...", "..."],
  "aspects": [
    {"entity": "...", "attribute": "...", "value": "...", "confidence": 0.95}
  ],
  "related_context": [
    {"text": "...", "type": "elaboration"}
  ],
  "contrastive_scope": ["...", "..."]
}"""


class AspectGenerator:
    """
    Generates structured aspects from text chunks using a Groq LLM.
    
    Args:
        groq_api_key: Your Groq API key (get free at console.groq.com)
        model: Groq model to use (llama-3.1-8b-instant is fast and free)
        temperature: Lower = more consistent/factual output
    """

    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
    ):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=temperature,
            max_tokens=1024,
        )
        self.model = model

    def generate(self, chunk_text: str, chunk_id: str = "") -> dict:
        """
        Generate aspects for a single chunk.
        
        Returns a dict with keys: summary, key_topics, aspects, related_context,
        contrastive_scope, raw_chunk, chunk_id, generation_error (if any)
        """
        messages = [
            SystemMessage(content=ASPECT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Extract aspects from this text chunk:\n\n---\n{chunk_text}\n---"
            ),
        ]

        result = {
            "chunk_id": chunk_id,
            "raw_chunk": chunk_text,
            "summary": "",
            "key_topics": [],
            "aspects": [],
            "related_context": [],
            "contrastive_scope": [],
            "generation_error": None,
        }

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Strip markdown fences if model ignores instructions
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            result.update({
                "summary": parsed.get("summary", ""),
                "key_topics": parsed.get("key_topics", []),
                "aspects": parsed.get("aspects", []),
                "related_context": parsed.get("related_context", []),
                "contrastive_scope": parsed.get("contrastive_scope", []),
            })

        except json.JSONDecodeError as e:
            result["generation_error"] = f"JSON parse error: {e}"
        except Exception as e:
            result["generation_error"] = f"LLM error: {e}"

        return result

    def generate_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Generate aspects for a list of chunks.
        Each chunk dict must have: 'text' and 'chunk_id'.
        
        Returns list of aspect dicts (same order as input).
        """
        results = []
        for i, chunk in enumerate(chunks):
            print(f"  [Aspect Generator] Chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
            aspect_data = self.generate(
                chunk_text=chunk["text"],
                chunk_id=chunk["chunk_id"]
            )
            results.append(aspect_data)
        return results


if __name__ == "__main__":
    import os
    KEY = os.environ.get("GROQ_API_KEY", "your-groq-key-here")

    gen = AspectGenerator(groq_api_key=KEY)

    test_chunk = """
    Transformer models use self-attention mechanisms to process sequences in parallel.
    Unlike RNNs, transformers do not process tokens sequentially, enabling much faster
    training on modern GPUs. The attention mechanism assigns weights to each token pair,
    allowing the model to capture long-range dependencies. BERT uses bidirectional attention,
    while GPT uses causal (left-to-right) attention.
    """

    result = gen.generate(test_chunk, chunk_id="test_001")

    if result["generation_error"]:
        print(f"ERROR: {result['generation_error']}")
    else:
        print(json.dumps(result, indent=2))