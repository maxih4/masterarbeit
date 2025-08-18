from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings



class ModelManager:
    def __init__(self, llm_model:BaseChatModel,embedding_model:Embeddings):
        """
        Initialize the ModelManager with a language model and an embedding model.
        :param llm_model: The language model to be used for generating embeddings.
        :param embedding_model: The embedding model to be used for generating sentence embeddings.
        """
        self.llm_model = llm_model
        self.embedding_model=embedding_model
