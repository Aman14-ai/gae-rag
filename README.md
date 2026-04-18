# GAE-RAG: Grounded Aspect Expansion RAG

A research implementation of a novel RAG pipeline that enriches document 
representations with verified aspects before indexing, enabling more precise
retrieval and grounded answer generation.

---

## Quick Start

### Step 1 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card needed)
3. Create an API key under **API Keys**

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

First-time install downloads ~200MB of local models (NLI + embedder).

### Step 3 — Set your API key

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

On Windows:
```bash
set GROQ_API_KEY=gsk_your_key_here
```

### Step 4 — Run the demo

```bash
cd gae_rag
python demo.py
```

---

## Project Structure

```
gae_rag/
├── pipeline.py                  ← Full end-to-end pipeline (start here)
├── demo.py                      ← Demo with sample docs + queries
├── requirements.txt
└── core/
    ├── aspect_generator.py      ← LLM-based aspect extraction
    ├── aspect_verifier.py       ← NLI-based aspect verification
    ├── chunker_embedder.py      ← Chunking + dual embedding
    ├── vector_store.py          ← ChromaDB dual-index store
    ├── retriever.py             ← Adaptive retrieval (3 stages)
    └── generator_grounding.py   ← Reranker + generator + grounding check
```

---

## Architecture

### Indexing Pipeline

```
Raw Documents
    │
    ▼
SemanticChunker          → splits on paragraphs/sentences, adds overlap
    │
    ▼
AspectGenerator (LLM)    → extracts entity-attribute pairs + related context
    │                       + contrastive scope tags (what's NOT covered)
    ▼
AspectVerifier (NLI)     → filters aspects below entailment threshold τ
    │                       Uses: cross-encoder/nli-deberta-v3-small
    ▼
DualEmbedder             → creates TWO vectors per chunk:
    │                       [1] original text embedding
    │                       [2] (original + aspects) embedding
    ▼
HybridVectorStore        → ChromaDB with TWO collections
                            + contrastive_scope stored as metadata
```

### Query Pipeline

```
User Query
    │
    ▼
AdaptiveRetriever
  Stage 1: Query both indexes, fuse scores
  Stage 2: If confidence < τ → query rewrite (LLM), retry
  Stage 3: Aspect-aware expansion using top results' topics
    │
    ▼
ContrastivePenalty       → penalize chunks whose scope doesn't match query
    │
    ▼
CrossEncoderReranker     → cross-encoder/ms-marco-MiniLM-L-6-v2
    │
    ▼
AnswerGenerator (LLM)    → grounded, citation-aware generation
    │
    ▼
GroundingChecker (NLI)   → extract claims → verify each against chunks
    │
    ├── Attribution rate ≥ threshold → ✓ Return answer
    │
    └── Below threshold → retry with focused question (up to N retries)
```

---

## Configuration Guide

### Key parameters in `GAERAGPipeline`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `nli_tau` | 0.5 | Aspect verification strictness. Lower → more aspects kept (more noise). Higher → fewer but more precise aspects. |
| `grounding_threshold` | 0.7 | Fraction of claims that must be grounded. Lower → more permissive. |
| `confidence_threshold` | 0.45 | Retrieval score below which query rewriting triggers. |
| `alpha` | 0.5 | Weight of aspect-enhanced index in score fusion. 0 = original only, 1 = aspect only. |
| `generation_model` | `llama-3.1-8b-instant` | Groq model for answers. Larger = better quality. |
| `utility_model` | `llama3-8b-8192` | For rewriting/extraction. Smaller = faster/cheaper. |

### Free Groq models available:

| Model | Speed | Quality | Best for |
|-------|-------|---------|----------|
| `llama3-8b-8192` | Very fast | Good | Rewriting, extraction |
| `llama-3.1-8b-instant` | Fast | Excellent | Answer generation |
| `mixtral-8x7b-32768` | Fast | Very good | Longer context |
| `gemma2-9b-it` | Fast | Good | Balanced |

---

## Using Your Own Documents

```python
from pipeline import GAERAGPipeline
import os

rag = GAERAGPipeline(groq_api_key=os.environ["GROQ_API_KEY"])

# From plain strings
docs = [
    {"text": "Your document text here...", "doc_id": "doc_001"},
    {"text": "Another document...", "doc_id": "doc_002",
     "metadata": {"source": "paper", "year": 2024}},
]

# From files
import glob
docs = []
for filepath in glob.glob("./papers/*.txt"):
    with open(filepath) as f:
        docs.append({
            "text": f.read(),
            "doc_id": filepath,
            "metadata": {"file": filepath}
        })

# Index once
rag.index(docs)

# Query many times (index persists to disk)
result = rag.query("What is the main contribution?")
print(result["answer"])

# Access retrieval details
for chunk in result["reranked_chunks"]:
    print(f"  [{chunk['chunk_id']}] score={chunk['reranker_score']:.3f}")
    print(f"  {chunk['text'][:100]}...")
```

---

## Research Notes

### Novel Contributions vs Prior Work

**1. Contrastive Scope Tags** (novel)  
Prior work on document expansion (e.g., DocT5Query, SPLADE) generates additional
query-like text but doesn't explicitly encode what a chunk *doesn't* cover.
The `contrastive_scope` field enables negative scoring — penalizing chunks that
explicitly exclude the query topic.

**2. NLI-Gated Aspect Indexing** (novel combination)  
Using NLI to verify generated aspects before indexing prevents hallucinated
aspects from corrupting the index. Prior aspect-based retrieval work (ARES, ASPECT)
doesn't include a verification gate.

**3. Dual Embeddings + Score Fusion** (extends prior work)  
Related to HyDE (Hypothetical Document Embeddings) but uses verified structured
aspects rather than hypothetical documents. The two-index architecture is
inspired by hybrid sparse-dense retrieval but applies semantic expansion rather
than lexical expansion.

**4. Grounding-Aware Retry Loop** (novel)  
Post-generation NLI verification with targeted retry is novel in RAG. RAGAS
measures faithfulness post-hoc but doesn't feed failures back into retrieval.
Self-RAG uses a "critique" token but is trained end-to-end; our approach is
training-free.

### Suggested Ablation Studies

For your paper's Table 4 / Table 5:

| System | Recall@5 | Faithfulness | AnswerRel |
|--------|----------|--------------|-----------|
| Vanilla RAG (baseline) | - | - | - |
| + Aspect expansion | - | - | - |
| + NLI verification | - | - | - |
| + Dual embeddings | - | - | - |
| + Adaptive retrieval | - | - | - |
| + Grounding retry (full GAE-RAG) | - | - | - |

### Evaluation Datasets

- **HotpotQA** — multi-hop, tests whether aspects help bridge reasoning chains
- **ASQA** — aggregated QA, tests faithfulness on ambiguous queries
- **QASPER** — scientific papers, tests aspect-targeted retrieval
- **RGB** — specifically tests RAG robustness and hallucination

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_groq'`**  
→ Run `pip install -r requirements.txt`

**`Error: GROQ_API_KEY not set`**  
→ Run `export GROQ_API_KEY="gsk_..."` before running

**NLI model download is slow**  
→ First run downloads ~180MB. Subsequent runs use cache in `~/.cache/huggingface/`

**ChromaDB data location**  
→ Default: `./gae_rag_db/`. Change `persist_dir` in `GAERAGPipeline()`

**Rate limit errors from Groq**  
→ Free tier: ~30 requests/min. Add `time.sleep(2)` between indexing steps
→ Or reduce batch size / use cached results

**Reranker loading is slow**  
→ Set `use_reranker=False` in `GAERAGPipeline()` for faster testing
