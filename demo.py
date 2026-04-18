"""
GAE-RAG Demo
=============
Demonstrates the full pipeline with sample RAG/ML research documents.
Run this after setup to verify everything works.

Usage:
    export GROQ_API_KEY="gsk_your_key_here"
    python demo.py
    
    # Or inline:
    GROQ_API_KEY=gsk_... python demo.py
"""

import os
import sys

# ── Config ──────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("ERROR: Set GROQ_API_KEY environment variable first.")
    print("  export GROQ_API_KEY='gsk_your_key_here'")
    sys.exit(1)

from pipeline import GAERAGPipeline

# ── Sample documents (RAG/ML research content) ──────────────────────────────
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "rag_survey_001",
        "metadata": {"source": "survey", "topic": "RAG foundations"},
        "text": """
Retrieval-Augmented Generation (RAG) is a framework that enhances large language models
by retrieving relevant documents from an external knowledge base before generating an answer.
Unlike pure generative models that rely on parametric memory, RAG grounds responses in 
retrieved evidence, reducing hallucinations and improving factual accuracy.

The standard RAG pipeline consists of three stages: indexing, retrieval, and generation.
During indexing, documents are split into chunks, embedded using dense encoders, and stored
in a vector database. At query time, the user's question is embedded and compared against
stored chunks using cosine similarity. The top-k most similar chunks are retrieved and
passed to the language model as context for answer generation.

Key limitations of vanilla RAG include semantic mismatch between query and document
phrasing, insufficient context in individual chunks, and lack of grounding verification.
Advanced RAG techniques address these through query rewriting, hypothetical document
embedding (HyDE), and multi-hop retrieval.
        """,
    },
    {
        "doc_id": "transformer_arch_002",
        "metadata": {"source": "textbook", "topic": "transformers"},
        "text": """
The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
revolutionized natural language processing by replacing recurrent networks with self-attention.
The architecture consists of stacked encoder and decoder blocks, each containing multi-head
attention and feed-forward sublayers with residual connections and layer normalization.

Self-attention computes a weighted sum of value vectors, where weights are determined by
the compatibility between query and key vectors. Multi-head attention runs multiple attention
operations in parallel, allowing the model to attend to different representation subspaces
simultaneously. The attention formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack
and is pre-trained with masked language modeling (MLM) and next sentence prediction (NSP).
GPT models use only the decoder with causal (autoregressive) attention. BERT's bidirectional
attention allows it to see context from both left and right, making it better for
classification and understanding tasks.
        """,
    },
    {
        "doc_id": "embeddings_003",
        "metadata": {"source": "paper", "topic": "dense retrieval"},
        "text": """
Dense passage retrieval (DPR) represents both queries and documents as dense vectors in
a shared embedding space, enabling semantic search beyond keyword matching. Unlike BM25
which relies on term frequency statistics, DPR can match "heart attack" with "myocardial
infarction" because they are semantically close in the embedding space.

Sentence transformers like all-MiniLM-L6-v2 produce 384-dimensional embeddings optimized
for semantic similarity. These models are trained with contrastive learning objectives where
similar sentence pairs are pulled closer in embedding space and dissimilar pairs are pushed apart.

Bi-encoder models (like DPR) embed query and document independently, enabling efficient
pre-computation of document embeddings. Cross-encoder models read query and document together
producing a more accurate relevance score but requiring inference at query time. The typical
RAG pipeline uses bi-encoders for retrieval and cross-encoders for reranking.
        """,
    },
    {
        "doc_id": "hallucination_004",
        "metadata": {"source": "paper", "topic": "hallucination and grounding"},
        "text": """
Hallucination in large language models refers to the generation of content that appears
plausible but is factually incorrect or unsupported by the provided context. In RAG systems,
hallucinations occur when the model relies on parametric memory rather than retrieved context,
or when retrieved context is insufficient or misleading.

Natural Language Inference (NLI) is a task to determine whether a hypothesis is entailed by,
neutral to, or contradicted by a given premise. NLI models like DeBERTa trained on NLI datasets
can be used to verify whether generated claims are supported by source documents. This approach,
known as grounding verification or attribution checking, computes an entailment score between
each generated claim and the retrieved passages.

RAGAS (Retrieval Augmented Generation Assessment) provides automatic evaluation metrics for
RAG systems including faithfulness (are claims grounded?), answer relevancy (does the answer
address the question?), context precision, and context recall. Faithfulness is computed by
dividing the number of grounded claims by total claims.
        """,
    },
    {
        "doc_id": "aspect_based_005",
        "metadata": {"source": "paper", "topic": "aspect-based NLP"},
        "text": """
Aspect-Based Sentiment Analysis (ABSA) decomposes document sentiment into entity-attribute
pairs called aspects. For example, in a restaurant review, aspects might include
(food, quality), (service, speed), (ambiance, noise_level). Each aspect captures what
is being evaluated (entity) and the dimension of evaluation (attribute).

Aspect extraction in NLP involves identifying opinion targets and their associated
properties from text. This structured representation captures more nuanced information
than document-level analysis. In the context of information retrieval, aspect-aware
indexing enriches document representations with explicit entity-attribute relationships,
enabling aspect-targeted retrieval — finding chunks that discuss specific facets of a topic.

Knowledge graph completion and aspect expansion share similar goals: enriching a sparse
representation with inferred or related knowledge while maintaining factual grounding.
Constrained generation with factual verification prevents the expansion from drifting
into hallucinated territory.
        """,
    },
    {
        "doc_id": "aspect_verification_006",
        "metadata": {"source": "paper", "topic": "aspect verification"},
        "text": """
Synthetic Dataset: “Behavioral Modulation Under Hierarchical Constraint
Systems”
Experimental Report: Dual-Class Agent System, N=600 Agents (120 Class-Ω, 480
Class-Σ)
Duration: 72-Hour Continuous Exposure + 24-Hour Recovery Phase
Initialization Layer
The experimental construct investigates behavioral drift in dual-class agent
systems operating under asymmetric authority gradients, with sample units
split across four geographically isolated clusters (Alpha, Beta, Gamma,
Delta). Two primary agent types are defined:
Class-Ω (Regulatory Units): 120 units (30 per cluster) programmed to enforce
hierarchical control via standardized signal protocols.
Class-Σ (Constrained Units): 480 units (120 Alpha, 180 Beta, 120 Gamma, 60
Delta control) with adaptive autonomy algorithms.
Initial baseline state (t₀) assumes equilibrium, where perceived autonomy (PA
= 0.6 ± 0.03) ≈ control pressure (CP = 0.58 ± 0.02). Deviations begin once
Environmental Intensity (EI) surpasses threshold parameter θ₁ = 4.2 EI units
(measured via signal noise, feedback latency, and resource scarcity proxies).
Note: Anomaly tracking relies on Event Marker Δ3 (breach event) and Protocol
Z-17 (environmental restructuring), with real-time data logged at 10-minute
intervals via automated behavioral coding software (accuracy = 89% cross
validated with human rater scores).
Variable Abstraction
The system defines a latent behavioral metric to quantify cumulative drift:
Behavioral Shift Index (BSI) = ∫₀^τ (EC × TA) d
"""
    }
]

# ── Test queries ──────────────────────────────────
QUERIES = [
    "What are the main limitations of vanilla RAG systems?",
    "How does BERT differ from GPT in terms of attention mechanism?",
    "What is the difference between bi-encoders and cross-encoders for retrieval?",
    "How can we detect hallucinations in RAG-generated answers?",
    "What are aspects in the context of information retrieval?",
    "In plain terms, who are the “rule-setters” and who are the “rule-followers” in this hierarchical experiment, and how is the authority difference built into the setup?"
]


def main():
    print("=" * 60)
    print(" GAE-RAG Demo")
    print("=" * 60)
    print()

    # Initialize pipeline
    rag = GAERAGPipeline(
        groq_api_key=GROQ_API_KEY,
        persist_dir="./gae_rag_demo_db",
        generation_model="llama-3.1-8b-instant",   # best quality for answers
        utility_model="llama-3.1-8b-instant",        # fast for rewriting/extraction
        nli_tau=0.45,                           # aspect verification threshold
        grounding_threshold=0.65,              # grounding check threshold
        confidence_threshold=0.40,             # retrieval confidence threshold
        use_reranker=True,
    )

    # Check if already indexed
    counts = rag.get_store_stats()
    if counts["original_index"] == 0:
        print("No existing index found. Indexing sample documents...")
        stats = rag.index(SAMPLE_DOCUMENTS)
    else:
        print(f"Found existing index with {counts['original_index']} chunks. Skipping re-indexing.")
        print("(To re-index, call rag.reset_index() first)\n")

    # Run queries
    print("\n" + "=" * 60)
    print(" RUNNING QUERIES")
    print("=" * 60)

    results_summary = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n\n[Query {i}/{len(QUERIES)}]")

        result = rag.query(q, k=4, alpha=0.5, verbose=True)

        results_summary.append({
            "query": q,
            "answer_preview": result["answer"][:200] + "...",
            "chunks_retrieved": len(result["reranked_chunks"]),
            "retries": result["retries"],
            "grounded": result["final_grounded"],
            "attribution_rate": result["grounding"].get("attribution_rate", 0.0),
        })

        # Pause between queries to respect rate limits
        import time
        time.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Query':<50} {'Grounded':<10} {'Attribution':<12} {'Retries'}")
    print("-" * 80)
    for r in results_summary:
        status = "✓" if r["grounded"] else "✗"
        q_short = r["query"][:48] + ".." if len(r["query"]) > 48 else r["query"]
        print(
            f"{q_short:<50} {status:<10} {r['attribution_rate']:.0%}{'':>8} {r['retries']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()