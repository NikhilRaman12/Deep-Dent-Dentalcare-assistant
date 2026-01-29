# =============================
# Deep-Dent RAG Gradio App
# =============================

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# STEP 1: Load PDF
# -----------------------------
loader = PyPDFLoader("Delivering_better_oral_health.pdf")
documents = loader.load()


# -----------------------------
# STEP 2: Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)


# -----------------------------
# STEP 3: Embeddings + Vector DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# STEP 4: Prompt
# -----------------------------
prompt = PromptTemplate.from_template(
    """You are Deep-Dent, a professional dental care assistant.

Use ONLY the provided context.
Do NOT mention documents, PDFs, or sources.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
)


# -----------------------------
# STEP 5: Local LLM (FREE)
# -----------------------------
llm = ChatOllama(
    model="tinyllama:latest",
    temperature=0
)


# -----------------------------
# STEP 6: LCEL RAG Chain
# -----------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# STEP 7: Gradio function
# -----------------------------
def ask_deep_dent(question):
    try:
        if not question.strip():
            return "Please ask a valid dental health question."
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Internal error: {str(e)}"

# -----------------------------
# STEP 8: Gradio UI
# -----------------------------
app = gr.Interface(
    fn=ask_deep_dent,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask a dental health question"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Deep-Dent Answer"
    ),
    title="Deep-Dent – Dental Health Assistant",
    description="Evidence-based dental care assistant powered by RAG and a local LLM"
   
)


# -----------------------------
# STEP 9: Launch
# -----------------------------
if __name__ == "__main__":
    app.launch(share=True)
