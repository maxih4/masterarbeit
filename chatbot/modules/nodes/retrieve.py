# Define steps
import asyncio
import logging
from typing import Dict, List

from module_instances import create_db_manager
from modules.rag.state import RetrieveState, QC

logger = logging.getLogger(__name__)


async def retrieve(state: RetrieveState) -> Dict[str, List[QC]]:

    db_manager = create_db_manager(drop_old=False)

    question = state["question"]
    # for each question form an asimilarity-search

    logger.info(state)
    result = await db_manager.vector_store.asimilarity_search(
        question,
        k=4,
        ranker_type="rrf",
        ranker_params={"k": 60},
        expr=_get_expression(state),
    )
    logger.info(result)
    return {"qc_pairs": [{"q": question, "ctx": result}]}


def _get_expression(state: RetrieveState):
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None
