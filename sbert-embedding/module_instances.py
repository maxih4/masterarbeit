from modules.database_manager import DatabaseManager
from modules.model_manager import ModelManager
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
        model_name="Qwen/Qwen3-Embedding-8B",
        model_kwargs={"trust_remote_code": True},
    ),
)


def create_db_manager(drop_old: bool = False) -> DatabaseManager:
    return DatabaseManager(
        vector_store=Milvus(
            embedding_function=model_manager.embedding_model,
            connection_args={"uri": "http://localhost:19530"},
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            auto_id=True,
            drop_old=drop_old,
            enable_dynamic_field=True,
            collection_name="async_test_collection",
        ),
        conn_pool=AsyncConnectionPool(
            conninfo="postgresql://postgres:example@localhost:5432/postgres?sslmode=disable",
            max_size=10,
            open=False,
        ),
    )
