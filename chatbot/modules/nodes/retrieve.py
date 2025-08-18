# Define steps
import asyncio
import logging
from typing import Dict, List

from module_instances import create_db_manager
from modules.rag.state import RetrieveState, QA

logger = logging.getLogger(__name__)


async def retrieve(state: RetrieveState) -> Dict[str, List[QA]]:

    # retrieved_docs = await db_manager.vector_store.asimilarity_search(
    #     state["questions"][0],
    #     k=3,
    #     ranker_type="rrf",
    #     ranker_params={"k": 100},
    #     expr=_get_expression(state),
    # )
    #
    # retrieved_docs_weighted = await db_manager.vector_store.asimilarity_search(
    #     state["questions"][0],
    #     k=3,
    #     ranker_type="weighted",
    #     ranker_params={"weights": [0.9, 0.1]},
    #     expr=_get_expression(state),
    # )
    # for doc in retrieved_docs:
    #     print("Doc metadata:", doc.metadata)
    #
    # print("Weighted Docs:")
    # for doc in retrieved_docs_weighted:
    #     print("Weighted Doc metadata:", doc.metadata)
    # return {"context": retrieved_docs}

    db_manager = create_db_manager(drop_old=False)

    question = state["question"]
    # for each qeustion form an asimilarity-search

    logger.info(state)
    result = await db_manager.vector_store.asimilarity_search(
        question,
        k=4,
        ranker_type="rrf",
        ranker_params={"k": 60},
        expr=_get_expression(state),
    )
    logger.info(result)
    return {"qa_pairs": [{"q": question, "ctx": result}]}


def _get_expression(state: RetrieveState):
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None
