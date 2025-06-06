import time
from fastapi.responses import StreamingResponse
from pymilvus import  WeightedRanker
from classes.ModelManager import ModelManager
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, Request
from transformers.pipelines import pipeline
from RagManager import graph




# res = model_manager.db.search(query_embedding)


app = FastAPI()
@app.get("/search")
async def search(sentence: str):
    """
    Search for similar sentences in the database.
    
    :param query: The query sentence to search for.
    :return: A list of similar sentences.
    """
    def stream_generator():
        for step in graph.stream({"question": sentence}, stream_mode="updates"):
            yield f"{step}\n\n----------------\n"

    
    return StreamingResponse(stream_generator(), media_type="text/plain")

###
# Middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response