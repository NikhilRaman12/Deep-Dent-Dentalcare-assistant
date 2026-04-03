"""
INTERVIEW PRESENTATION GUIDE
============================

What to explain:

1. PROBLEM STATEMENT (30 seconds)
   "I built a Retrieval-Augmented Generation (RAG) system for dental health queries.
    The challenge is: how do you make an LLM provide accurate, grounded answers
    instead of hallucinating? Solution: retrieve relevant documents FIRST,
    then ask the LLM to answer based only on those documents."

2. ARCHITECTURE (1 minute)
   - Load PDF → Takes dental care document as input
   - Chunk → Split into 1000-char chunks (with 100 overlap) for semantic meaning
   - Embed → Convert chunks to vectors using sentence-transformers (384-dim)
   - Store → FAISS vector index for fast similarity search
   - Retrieve → Find top-4 most relevant chunks for each query
   - Prompt → Pass context + question to LLM
   - Generate → Local Ollama model produces grounded answer

3. CODE ORGANIZATION (1 minute)
   Show the file structure:
   ✓ data_loader.py     → 5 lines, loads PDF
   ✓ chunking.py        → 7 lines, splits into chunks
   ✓ vector_store.py    → 10 lines, embeddings + retrieval
   ✓ generator.py       → 20 lines, prompt + LLM + RAG chain
   ✓ main.py            → 45 lines, CLI demo
   ✓ app.py             → 50 lines, Gradio UI
   ✓ evaluation.py      → 40 lines, benchmarking
   ✓ monitoring.py      → 15 lines, Prometheus metrics
   
4. KEY METRICS (1 minute)
   Show run_evaluation.py output:
   - Latency: ~1.2s avg (acceptable for healthcare)
   - Token cost: ~45 tokens per response (cheap + fast)
   - Precision@4: Retrieved docs are relevant
   - Works offline (no API calls)

5. LIVE DEMO (2 minutes)
   Run: python main.py
   Show 3 queries with latency + tokens
   
6. BUSINESS IMPACT (1 minute)
   - Accuracy: Grounds responses in medical documents
   - Compliance: No hallucinations, auditable answers
   - Cost: Free local LLM (no API calls)
   - Scalability: Can handle multiple users

TALKING POINTS FOR QUESTIONS:

Q: How would you handle fine-tuning?
A: "I'd collect Q&A pairs from dental experts, fine-tune the embedding model
   on dental-specific terminology, or fine-tune the LLM on dental answers.
   Trade-off: accuracy vs. latency."

Q: How to handle out-of-domain queries?
A: "Add a confidence threshold. If retrieval score < 0.3, say 'Not in knowledge base'.
   Track these for expert review."

Q: Scaling to 1M users?
A: "1. Cache vector embeddings
    2. Use distributed FAISS (Facebook)
    3. Add LLM serving layer (vLLM/Ray)
    4. Collect metrics with Prometheus + Grafana"

Q: How to evaluate quality?
A: "Ground truth evaluation: 50 Q&A pairs from dental experts,
    compare RAG answer vs. expert using BLEU/ROUGE + human review."
"""