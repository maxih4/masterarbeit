from pymilvus import AnnSearchRequest, MilvusClient, DataType, WeightedRanker
from .Sentence import Sentence


class DatabaseManager:
    def __init__(self, client: MilvusClient, ranker:WeightedRanker, collectionName: str = "my_collection2"):
        self.client=client
        self.collection_name = collectionName
        schema = self.__create_and_fill_schema()
        index_params = self.__set_index_params()
        self.__create_collection(schema=schema, index_params=index_params)
        self.ranker = ranker


    def __create_and_fill_schema(self):
        """
        Create a schema for the database collection.
        """
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)

        return schema
    

    def __set_index_params(self):
        """
        Set index parameters for vector fields.
        
        :return: Index parameters.
        """
        index_params = self.client.prepare_index_params()
        
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        
        return index_params    
    
    def __create_collection(self, schema, index_params):
        """
        Create a collection in the database.
        
        :param collection_name: Name of the collection to create.
        :param schema: Schema for the collection.
        :param index_params: Index parameters for the collection.
        """
        if self.client.has_collection(collection_name=self.collection_name) == False:
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )        


    def insert_sentence(self,sentence:Sentence):
        self.client.insert(collection_name=self.collection_name,
                             data={
                         "dense_vector": sentence.dense_vector,
                         "sparse_vector": sentence.sparse_vector,
                         "text": sentence.text
                    })
    
    def search(self,query_sentence:Sentence, limit: int = 2):
        """
        Search for similar sentences in the database.
        
        :param query_sentence: The sentence to search for.
        :param limit: The maximum number of results to return.
        :return: A list of similar sentences.
        """
        search_param_1 = {
        "data": [query_sentence.dense_vector],
        "anns_field": "dense_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        },
        "limit": limit
        }

        search_param_2 = {
        "data": [query_sentence.sparse_vector],
        "anns_field": "sparse_vector",
        "param": {
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2}
        },
        "limit": limit
         }
        
        request_1 = AnnSearchRequest(**search_param_1)
        request_2 = AnnSearchRequest(**search_param_2)

        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[request_1, request_2],
            ranker=self.ranker,
            limit=limit,
            output_fields=["text"]
        )