from langchain_core.vectorstores.base import VectorStore



class DatabaseManager:
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the DatabaseManager with a vector store.
        :param vector_store: The vector store to be used for storing and retrieving sentence embeddings.
        """
        self.vector_store= vector_store

