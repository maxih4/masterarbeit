
from pymilvus import MilvusClient, WeightedRanker
from .DatabaseManager import DatabaseManager
from FlagEmbedding import BGEM3FlagModel
from .Sentence import Sentence


class ModelManager:
    def __init__(self, model:BGEM3FlagModel, database_url: str, ranker: WeightedRanker):
        """
        Initialize the EmbeddingManager with a model and database connection.
        
        :param model: The embedding model to use.
        :param database_url: The URL of the database to connect to.
        """
        self.model = model
        self.database_url = database_url
        self.ranker=ranker
        self.db=self.__connect_to_database()


    def generate_vector_and_save(self, sentence: str):
        """
        Generate a vector for the given sentence and store it in the database.
        
        :param sentence: The sentence to be embedded.
        """

        # Generate the dense and sparse vectors using the model
        sentence_done = self.generate_embeddings(sentence)
        # Store the sentence along with its vectors in the database
        self.__store_sentence(sentence_done)


    def generate_embeddings(self,sentence:str)->Sentence:
        """
        Generate embeddings for the given sentence.
        
        :param sentence: The sentence to be embedded.
        :return: A Sentence object containing the text and its vectors.
        """
        embeddings = self.model.encode(sentence, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return Sentence(sentence, embeddings["lexical_weights"], embeddings["dense_vecs"])



    def __store_sentence(self, sentence: Sentence):
        """
        Store the generated vectors in the database.
        
        :param sentence: The sentence object containing the text and its vectors.
        """
        self.db.insert_sentence(sentence)

    def __connect_to_database(self):
        """
        Connect to the database using the provided URL.
        
        :return: A MilvusClient instance connected to the database.
        """
        return  DatabaseManager(MilvusClient(self.database_url),self.ranker)
