import logging
from fastapi import FastAPI
import uvicorn

from modules.rag_manager import RagManager
from langchain_core.runnables import RunnableConfig

from utils.logging_config import configure_logging

# ---------------------------------------------------
# Configure logging
# ---------------------------------------------------
configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# FastAPI setup
# ---------------------------------------------------
app = FastAPI()
ragManager = RagManager()


@app.get("/search")
async def search(sentence: str, thread_id: int):
    logger.info(f"Received search request: {sentence}")
    # Pass thread id
    config = RunnableConfig(configurable={"thread_id": thread_id})
    # Create graph
    graph = await ragManager.create_graph()
    if graph is None:
        logger.error("Graph is not initialized")
        raise Exception("Graph is not initialized")

    # Invoke graph with empty state and user input
    result = await graph.ainvoke(
        {"user_input": sentence, "qc_pairs": [], "token_usage": [], "input_tokens": 0},
        config,
    )
    # Return answer
    return result


# ---------------------------------------------------
# Main method
# ---------------------------------------------------
def main():
    logger.info("Starting FastAPI app…")
    uvicorn.run(
        "main:app",  # <--- import path: file name (main.py) : app
        host="0.0.0.0",
        port=8000,
        reload=True,  # optional: only for dev
    )


if __name__ == "__main__":
    main()
