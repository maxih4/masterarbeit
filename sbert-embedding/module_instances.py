from modules.DatabaseManager import DatabaseManager
from modules.ModelManager import ModelManager
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from psycopg_pool import AsyncConnectionPool


model_manager = ModelManager(
    llm_model=ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
    ),
    embedding_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ),
)

db_manager = DatabaseManager(
    vector_store=Milvus(
        embedding_function=model_manager.embedding_model,
        connection_args={"uri": "http://localhost:19530"},
        builtin_function=BM25BuiltInFunction(),
        # `dense` is for OpenAI embeddings, `sparse` is the output field of BM25 function
        vector_field=["dense", "sparse"],
        auto_id=True,
    ),
    conn_pool=AsyncConnectionPool(
        conninfo="postgresql://postgres:example@localhost:5432/postgres?sslmode=disable",
        max_size=10,
        open=False,
    ),
)


