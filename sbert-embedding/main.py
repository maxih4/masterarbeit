
#
# @app.get("/search")
# async def search(sentence: str):
#     config = RunnableConfig(
#         configurable={
#             "thread_id": "1"
#         }
#     )
#     graph =await ragManager.create_graph()
#     if(graph is None):
#         raise Exception("Graph is not initialized")
#     result = await graph.ainvoke({"user_input": sentence}, config)
#     return result
#


import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from modules.RagManager import RagManager
from langchain_core.runnables import RunnableConfig

# ---------------------------------------------------
# Configure logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# FastAPI setup
# ---------------------------------------------------
app = FastAPI()
ragManager = RagManager()


@app.get("/search")
async def search(sentence: str):
    logger.info(f"Received search request: {sentence}")
    config = RunnableConfig(
        configurable={"thread_id": "1"}
    )
    graph = await ragManager.create_graph()
    if graph is None:
        logger.error("Graph is not initialized")
        raise Exception("Graph is not initialized")

    async def event_stream():
        async for update in graph.astream({"user_input": sentence}, config, stream_mode="updates"):
            logger.debug(f"Streaming update: {update}")
            yield f"data: {update}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------
# Main method
# ---------------------------------------------------
def main():
    logger.info("Starting FastAPI app…")
    uvicorn.run(
        "main:app",  # <--- import path: file name (main.py) : app
        host="0.0.0.0",
        port=8000,
        reload=True  # optional: only for dev
    )


if __name__ == "__main__":
    main()
