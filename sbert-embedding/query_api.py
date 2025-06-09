import time
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from RagManager import RagManager
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.postgres import PostgresSaver


# res = model_manager.db.search(query_embedding)


app = FastAPI()
ragManager = RagManager()



@app.get("/search")
async def search(sentence: str):
    config = RunnableConfig(
        configurable={
            "thread_id": "1"
        }
    )
    graph =await ragManager.create_graph()
    if(graph is None):
        raise Exception("Graph is not initialized")
    result = await graph.ainvoke({"user_input": sentence}, config)
    return result
