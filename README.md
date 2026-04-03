\# Deep-Dent RAG Assistant

Production-grade Retrieval-Augmented Generation system for dental health queries using local LLM.

## Architecture

```
PDF Document
    ↓
  Chunks (1000 chars, 100 overlap)
    ↓
  Embeddings (all-MiniLM-L6-v2)
    ↓
  FAISS Vector Store
    ↓
  Retriever (top-4 docs)
    ↓
  LangChain LCEL Pipeline
    ↓
  Ollama LLM (tinyllama)
    ↓
  Response
```

## Key Features

✅ **Modular Design**: Each component in separate file  
✅ **Interview Ready**: Clean, simple code  
✅ **Metrics**: Latency, token cost, precision@k, recall  
✅ **Monitoring**: Prometheus-ready  
✅ **Free**: Uses local Ollama LLM  

## Project Structure

```
├── data_loader.py       # PDF loading
├── chunking.py          # Document chunking
├── vector_store.py      # Embeddings + FAISS
├── generator.py         # Prompt + LLM + RAG chain
├── main.py              # CLI demo
├── app.py               # Gradio web UI
├── evaluation.py        # Benchmarking
├── run_evaluation.py    # Eval runner
├── monitoring.py        # Prometheus metrics
├── requirements.txt     # Dependencies
└── Delivering_better_oral_health.pdf
```

## Quick Start

```bash
# 1. Clone & setup
git clone <repo>
cd Deep-Dent-Dentalcare-assistant
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install
pip install -r requirements.txt

# 3. Run CLI demo
python main.py

# 4. Run web UI
python app.py

# 5. Run evaluation
python run_evaluation.py
```

## Evaluation Metrics

- **Precision@K**: % relevant in top-k results
- **Recall**: % of relevant docs retrieved
- **Latency**: Response time (milliseconds)
- **Token Cost**: Tokens per response

## Configuration

Edit in respective files:
- Chunk size: `chunking.py`
- K (retrieval): `vector_store.py`
- Model: `generator.py`
- Prompt: `generator.py`

## Business Context

**Problem**: Generic chatbots give unreliable health information  
**Solution**: Ground responses in authoritative dental documents  
**Benefit**: Accurate, consistent, compliant answers at scale  

---

*Interview-ready production RAG system for healthcare domain.*



