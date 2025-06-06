from classes.DatabaseManager import DatabaseManager
from classes.ModelManager import ModelManager
from classes.InputManager import FAQInputManager
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus,BM25BuiltInFunction


model_manager = ModelManager(
    llm_model=ChatOllama(
        model="llama3.2",
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
        vector_field=["dense", "sparse"],
        auto_id=True,
        drop_old=True
    )
)


faq_input_manager = FAQInputManager()