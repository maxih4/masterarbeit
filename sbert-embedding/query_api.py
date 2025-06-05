import time
from pymilvus import  WeightedRanker
from classes.ModelManager import ModelManager
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, Request
from transformers.pipelines import pipeline




model_manager = ModelManager(model=BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True), database_url='http://localhost:19530', ranker=WeightedRanker(0.9,0.1))
pipe = pipeline("text2text-generation", model="oliverguhr/spelling-correction-german-base")     

# query="Kann ich einen bestellten Container abbestellen?"
# query_embedding = model_manager.generate_embeddings(query)
# res = model_manager.db.search(query_embedding)

# for hits in res:
#     print("TopK results:")
#     for hit in hits:
#         print(hit)
app = FastAPI()
@app.get("/search")
async def search(sentence: str):
    """
    Search for similar sentences in the database.
    
    :param query: The query sentence to search for.
    :return: A list of similar sentences.
    """
    corrected_sentence= pipe
    print(corrected_sentence)

    query_embedding = model_manager.generate_embeddings(sentence)
    res = model_manager.db.search(query_embedding, limit=5)
    
    results = []
    for hits in res:
        for hit in hits:
            results.append(hit.text) # type: ignore
    
    return {"results": results}

###
# Middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response