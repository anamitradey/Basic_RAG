from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

embedder = SentenceTransformerEmbeddings(
              model_name="sentence-transformers/all-MiniLM-L6-v2",
              encode_kwargs={"normalize_embeddings": False})   # or True, but be consistent

vectordb = Chroma(
    persist_directory="./db",
    embedding_function=embedder,
)
q = "What colour is  Car"
print(vectordb.similarity_search(q, k=5)[1].page_content[:150])

