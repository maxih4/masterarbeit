from langchain_core.vectorstores.base import VectorStore
from psycopg_pool import AsyncConnectionPool





class DatabaseManager:
    def __init__(self, vector_store: VectorStore, conn_pool: AsyncConnectionPool):
        """
        Initialize the DatabaseManager with a vector store.
        :param vector_store: The vector store to be used for storing and retrieving sentence embeddings.
        """
        self.vector_store= vector_store
        self.conn_pool = conn_pool

