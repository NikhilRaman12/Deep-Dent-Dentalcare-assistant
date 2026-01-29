# -----------------------------
# Deep-Dent RAG Chatbot (Stable)
# -----------------------------

# STEP 1: Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



# STEP 2: Load PDF
loader = PyPDFLoader("Delivering_better_oral_health.pdf")
documents = loader.load()
print("Pages loaded:", len(documents))


# STEP 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print("Chunks created:", len(chunks))


# STEP 4: Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)


# STEP 5: Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# STEP 6: Prompt
prompt = PromptTemplate.from_template(
    """
You are Deep-Dent, a professional dental care assistant.
Answer strictly using the provided context.

Rules:
- Do NOT mention documents, metadata, or internal objects
- Do NOT quote raw text blocks
- Respond as a clear professional summary or step by step reasoning based on the user question
- No blank statements
- You are not replacing surgical dentist
- You are here to nourish and protect the teeth

Context:
{context}

Question:
{question}

Final Answer:
"""
)




# STEP 7: Local LLM (FREE)
llm = ChatOllama(
    model="tinyllama:latest",
    temperature=0
)


# STEP 8: RAG Pipeline (LCEL – Modern)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# STEP 9: Query
query = "What are the most effective practices to protect teeth from decay?"
response = rag_chain.invoke(query)


# STEP 10: Output
print("\nDeep-Dent Answer:\n")
print(response)
