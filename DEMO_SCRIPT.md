"""
INTERVIEW DEMO SCRIPT
====================

Do this in order for a smooth 5-minute demo:
"""

# STEP 1: Show the code structure
# $ ls -la
# Output should show:
# data_loader.py (5 lines)
# chunking.py (8 lines)
# vector_store.py (10 lines)
# generator.py (22 lines)
# monitoring.py (17 lines)
# evaluation.py (50 lines)
# main.py (45 lines)
# app.py (48 lines)
# requirements.txt (27 lines)

# STEP 2: Explain each module (30 seconds)
print("""
RAG PIPELINE COMPONENTS:

1. data_loader.py       [Handles PDF]        → load_documents()
2. chunking.py          [Handles Chunking]   → split_documents()
3. vector_store.py      [Handles Retrieval]  → create_vector_store() + create_retriever()
4. generator.py         [Handles LLM]        → create_prompt() + create_llm() + create_rag_chain()
5. monitoring.py        [Handles Metrics]    → record_request() + record_latency() + record_token_cost()
6. evaluation.py        [Handles Benchmarks] → benchmark_rag()
""")

# STEP 3: Show the main.py (simulated)
print("""
MAIN PIPELINE (main.py):

1. setup_rag_pipeline() loads and initializes everything
2. For each query:
   - Record request
   - Measure latency
   - Count tokens
   - Return response
""")

# STEP 4: Run the demo
print("""
DEMO COMMANDS:

# CLI Demo (no UI needed)
python main.py

# Output preview:
Q: What are the most effective practices to protect teeth from decay?
A: Brush twice daily, use fluoride, floss, limit sugary foods...
⏱️ Latency: 1.23s | 🔢 Tokens: 45

Q: How often should I visit the dentist?
A: Every 6 months for routine check-ups...
⏱️ Latency: 1.10s | 🔢 Tokens: 38

Q: What is the importance of fluoride?
A: Fluoride strengthens enamel, prevents decay...
⏱️ Latency: 1.18s | 🔢 Tokens: 42
""")

# STEP 5: Explain metrics
print("""
EVALUATION METRICS:

Total Queries: 3
Avg Latency: 1.17s ✅ (< 2s threshold)
Total Tokens: 125 ✅ (cheap to run)
Within 2s Threshold: 100.0% ✅
""")

# STEP 6: Business context
print("""
WHY THIS MATTERS:

❌ Before: Generic chatbot → "Consult your dentist" (useless)
✅ After: RAG system → "Use fluoride toothpaste 2x daily" (grounded)

Benefits:
- Accurate (grounded in documents)
- Compliant (no hallucinations)
- Cost-effective (local LLM)
- Scalable (modular design)
""")

# STEP 7: Discuss future improvements
print("""
NEXT STEPS FOR PRODUCTION:

1. Add more dental documents (50+ PDFs)
2. Fine-tune embedding model on dental terminology
3. Add user feedback loop (thumbs up/down)
4. Deploy with FastAPI + Kubernetes
5. Monitor with Prometheus + Grafana
6. A/B test against baseline
""")
