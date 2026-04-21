import os
from pipeline import GAERAGPipeline

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

rag = GAERAGPipeline(
    groq_api_key=GROQ_API_KEY,
    persist_dir="./gae_rag_demo_db",
    generation_model="llama-3.1-8b-instant",
    utility_model="llama-3.1-8b-instant",
    nli_tau=0.45,
    grounding_threshold=0.65,
    confidence_threshold=0.40,
    use_reranker=True,
)

def ask():
    print("\n=== GAE-RAG Assistant ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        result = rag.query(query, k=5, alpha=0.5, verbose=False)

        print("\nAssistant:")
        print(result["answer"])

        print("\n--- Debug Info ---")
        print(f"Grounded: {result['final_grounded']}")
        print(f"Attribution: {result['grounding'].get('attribution_rate', 0):.2f}")
        print(f"Retries: {result['retries']}")
        print("------------------\n")


if __name__ == "__main__":
    ask()