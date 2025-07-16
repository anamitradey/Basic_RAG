# --- 0) Monkey‑patch sqlite so Chroma stops rejecting UBI 9's old 3.34 build ----
import pysqlite3, sys
sys.modules["sqlite3"] = pysqlite3

# --- 1) Std lib ---------------------------------------------------------------
import os
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import Body

from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain.embeddings import (
    OpenAIEmbeddings,
    CohereEmbeddings,
    HuggingFaceInstructEmbeddings,
)
# Optional local embedding model
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None
# Vector stores
from langchain.vectorstores import Chroma
try:
    from langchain.vectorstores import Pinecone
except ImportError:
    Pinecone = None
try:
    from langchain.vectorstores import Weaviate
except ImportError:
    Weaviate = None
# LLMs
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI, OpenAI

from langchain.llms import Anthropic, Cohere, HuggingFaceHub
# Optional local LLM
try:
    from langchain.llms import LlamaCpp
except ImportError:
    LlamaCpp = None
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment and config
def load_config(path: str = None) -> dict:
    load_dotenv()
    cfg_file = path or os.getenv("CONFIG_PATH", "config.yaml")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with open(cfg_file) as f:
        return yaml.safe_load(f)

cfg = load_config()

# --- Dynamic class maps ---
LLM_CLASSES = {
    "openai": OpenAI,
    "openai_chat": ChatOpenAI,
    "anthropic": Anthropic,
    "cohere": Cohere,
    "hf": HuggingFaceHub,
    "llamacpp": LlamaCpp,
}
EMBEDDER_CLASSES = {
    "openai": OpenAIEmbeddings,
    "cohere": CohereEmbeddings,
    "hf_instruct": HuggingFaceInstructEmbeddings,
    "hf_local": HuggingFaceEmbeddings,
}
VECTOR_STORE_CLASSES = {"chroma": Chroma, "pinecone": Pinecone, "weaviate": Weaviate}   

# Instantiate embedder
embed_key = cfg.get("embedder_provider", "openai")
EmbedClass = EMBEDDER_CLASSES.get(embed_key)
if EmbedClass is None:
    raise ImportError(f"Embedder '{embed_key}' not supported or missing dependencies.")
embed_opts = cfg.get("embedder_opts", {})
embedder = EmbedClass(**embed_opts)

# Instantiate vector store
vs_key = cfg.get("vector_store", "chroma")
VSClass = VECTOR_STORE_CLASSES.get(vs_key)
if VSClass is None:
    raise ImportError(
        f"Vector store '{vs_key}' unavailable. "
        "If using Pinecone or Weaviate, install langchain-community: `pip install git+https://github.com/langchain-ai/langchain-community`"
    )
vs_opts = cfg.get("vector_store_opts", {})
db_path = os.getenv("VECTOR_STORE_PATH", vs_opts.get("persist_directory", "./db"))
os.makedirs(db_path, exist_ok=True)
vectordb = VSClass(
    persist_directory=db_path,
    embedding_function=embedder,
    **vs_opts
)

# Setup chunker & retrieval
top_k = cfg.get("top_k", 5)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.get("chunk_size", 500), chunk_overlap=cfg.get("chunk_overlap", 50)
)

# Instantiate LLM
llm_key = cfg.get("llm_provider", "openai_chat")
LLMClass = LLM_CLASSES.get(llm_key)
if LLMClass is None:
    raise ImportError(f"LLM provider '{llm_key}' not supported or missing dependencies.")
llm_opts = cfg.get("llm_opts", {})
# For LlamaCpp, expect 'model_path' in llm_opts
llm: LLM = LLMClass(**llm_opts)

# Build QA chain
auto_retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type=cfg.get("chain_type", "stuff"),
    retriever=auto_retriever
)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    """Answer user question using the RAG QA chain."""
    return {"answer": qa_chain.run(q.question)}

class IngestPayload(BaseModel):
    texts: list[str]

@app.post("/ingest")
def ingest(payload: IngestPayload):
    """Ingest a list of raw text strings into the vector store."""
    if not payload.texts:
        raise HTTPException(status_code=400, detail="No texts provided for ingestion.")
    docs = [Document(page_content=txt) for txt in payload.texts]
    chunks = splitter.split_documents(docs)
    vectordb.add_documents(chunks)
    vectordb.persist()
    return {"ingested_chunks": len(chunks)}

@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a UTF-8 text file and ingest its content into the vector store."""
    # Validate size/types if desired
    text_chunks = []
    async for chunk in file.stream(4096):
        text_chunks.append(chunk.decode(errors='ignore'))
    docs = [Document(page_content="".join(text_chunks))]
    chunks = splitter.split_documents(docs)
    vectordb.add_documents(chunks)
    vectordb.persist()
    return {"filename": file.filename, "ingested_chunks": len(chunks)}

# NOTE: Deploy via external ASGI server (e.g., Uvicorn/Gunicorn); no in-code runner.
@app.post("/ingest-text")
async def ingest_text(raw_text: str = Body(..., media_type="text/plain")):
    """Ingest raw UTF-8 text directly (text/plain) without JSON quoting."""
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text provided for ingestion.")
    docs = [Document(page_content=raw_text)]
    chunks = splitter.split_documents(docs)
    vectordb.add_documents(chunks)
    vectordb.persist()
    return {"ingested_chunks": len(chunks)}