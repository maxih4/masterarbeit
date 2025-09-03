import os

from dotenv import load_dotenv
from langchain_experimental.data_anonymizer import PresidioAnonymizer
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
    AzureChatOpenAI,
)
from presidio_anonymizer import OperatorConfig

from modules.anonymizer_manager import AnonymizerManager
from modules.database_manager import DatabaseManager
from modules.model_manager import ModelManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from psycopg_pool import AsyncConnectionPool

load_dotenv()

model_manager = ModelManager(
    llm_model_classify=AzureChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME_CLASSIFY"),
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY_CLASSIFY"),
        azure_endpoint=os.environ.get("OPENAI_API_ENDPOINT_CLASSIFY"),
        azure_deployment=os.environ.get("OPENAI_API_DEPLOYMENT_CLASSIFY"),
        api_version=os.environ.get("OPENAI_API_VERSION_CLASSIFY"),
        max_tokens=None,
    ),
    # embedding_model=HuggingFaceEmbeddings(
    #     model_name="Qwen/Qwen3-Embedding-8B",
    #     model_kwargs={"trust_remote_code": True},
    # ),
    # embedding_model=HuggingFaceEmbeddings(
    #    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # ),
    embedding_model=AzureOpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL_NAME"),
        azure_endpoint=os.environ.get("EMBEDDING_MODEL_API_ENDPOINT"),
        api_key=os.environ.get("EMBEDDING_MODEL_API_KEY"),
        azure_deployment=os.environ.get("EMBEDDING_MODEL_DEPLOYMENT"),
    ),
    llm_model=AzureChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("OPENAI_API_ENDPOINT"),
        azure_deployment=os.environ.get("OPENAI_API_DEPLOYMENT"),
        api_version=os.environ.get("OPENAI_API_VERSION"),
        max_tokens=None,
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
            collection_name=os.environ.get("VECTOR_COLLECTION_NAME"),
        ),
        conn_pool=AsyncConnectionPool(
            conninfo="postgresql://postgres:example@localhost:5432/postgres?sslmode=disable",
            max_size=10,
            open=False,
        ),
    )


anonymizer_manager = AnonymizerManager(
    anonymizer=PresidioAnonymizer(
        analyzed_fields=["PHONE_NUMBER", "LOCATION", "EMAIL_ADDRESS", "PERSON"],
        languages_config={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "de", "model_name": "de_core_news_lg"},
            ],
        },
        operators={
            "PHONE_NUMBER": OperatorConfig(
                "replace", {"new_value": "<ANONYMIZED_PHONE>"}
            ),
            "LOCATION": OperatorConfig(
                "replace", {"new_value": "<ANONYMIZED_LOCATION>"}
            ),
            "EMAIL_ADDRESS": OperatorConfig(
                "replace", {"new_value": "<ANONYMIZED_MAIL>"}
            ),
            "PERSON": OperatorConfig("replace", {"new_value": "<ANONYMIZED_PERSON>"}),
        },
    )
)
