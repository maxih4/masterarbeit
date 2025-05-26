
# # 1. Connect to Milvus
# client = MilvusClient("http://localhost:19530")



# # 2. Create schema for database collection
# schema = client.create_schema(
#     auto_id=True,
#     enable_dynamic_fields=True,
# )

# #3. Add Fields to the schema
# schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
# schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
# schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

# #4. Set index params for vector fields
# index_params = client.prepare_index_params()

# index_params.add_index(
#     field_name="dense_vector",
#     index_name="dense_vector_index",
#     index_type="AUTOINDEX",
#     metric_type="COSINE"
# )

# index_params.add_index(
#     field_name="sparse_vector",
#     index_name="sparse_inverted_index",
#     index_type="SPARSE_INVERTED_INDEX",
#     metric_type="IP",
# )

# #5. Create a collection
# if client.has_collection(collection_name="my_collection"):
#     client.drop_collection(collection_name="my_collection")
# client.create_collection(
#     collection_name="my_collection",
#     schema=schema,
#     index_params=index_params
# )

# #6. Load the model
# model2 = BGEM3FlagModel('BAAI/bge-m3',  
#                         use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


# # The sentences to encode
# sentences = [
#      "Ich benötige einen häufigeren Turnus als auswählbar, was muss ich machen? : Für einen häufigeren Turnus setze dich bitte mit unserem Kundenservice in Verbindung.",
#      "It's so sunny outside!",
#      "He drove to the stadium.",
#  ]

# # #7. Encode sentences to get their embeddings
# output_1 = model2.encode(sentences[0], return_dense=True, return_sparse=True, return_colbert_vecs=False)
# # output_2 = model2.encode(sentences[1], return_dense=True, return_sparse=True, return_colbert_vecs=False)
# # output_3 = model2.encode(sentences[2], return_dense=True, return_sparse=True, return_colbert_vecs=False)
# print("Hier sind die embeddings:")
# dense_vector_1=output_1["dense_vecs"]
# print(type(dense_vector_1))



# #8. Store the embeddings in Milvus
# client.insert(
#     collection_name="my_collection",
#     data=[{"dense_vector":output_1["dense_vecs"],"sparse_vector":output_1["lexical_weights"]},
#            {"dense_vector":output_2["dense_vecs"],"sparse_vector":output_2["lexical_weights"]},
#            {"dense_vector":output_3["dense_vecs"],"sparse_vector":output_3["lexical_weights"]}],
# )


import os

from pymilvus import  WeightedRanker
from classes.InputManager import FAQInputManager
from classes.ModelManager import ModelManager
from FlagEmbedding import BGEM3FlagModel

faq_input_manager = FAQInputManager()
model_manager = ModelManager(model=BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True), database_url='http://localhost:19530', ranker=WeightedRanker(0.8,0.3))

# Get the directory where main.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "csv", "faq.csv")


faq_sentences = faq_input_manager.extract_sentences_from_csv(csv_path)
for sentence in faq_sentences:
  model_manager.generate_vector_and_save(sentence) 

######## Bis hierhin local, danach als fastapi auf dem server


# query="Kann ich einen bestellten Container abbestellen?"
# query_embedding = model_manager.generate_embeddings(query)

# res = model_manager.db.search(query_embedding)

# for hits in res:
#     print("TopK results:")
#     for hit in hits:
#         print(hit)

