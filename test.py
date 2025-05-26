from pymilvus import MilvusClient, Collection


db  = MilvusClient(
    "http://localhost:19530"
)

print(db.get("my_collection2",ids=[458298330813190589,458298330813190645],output_fields=["text"]))


