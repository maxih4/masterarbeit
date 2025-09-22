import logging
from typing import Dict, List

from module_instances import create_db_manager
from modules.rag.state import RetrieveState, QC

logger = logging.getLogger(__name__)


async def retrieve(state: RetrieveState) -> Dict[str, List[QC]]:
    """
    Retrieve context documents for a given user question.
    :param state: current graph state with classifier + question
    :return: dict with question-context pairs for downstream nodes
    """
    # Create db manager instance
    db_manager = create_db_manager(drop_old=False)

    # Extract question from state
    question = state["question"]
    classifier = state["classifier"]

    logger.info(f"[Retrieve] Processing question: {question}")
    logger.debug(f"[Retrieve] Classifier: {classifier}")

    # Build filter expression for vector store search
    expr = _get_expression(state)
    if expr:
        logger.debug(f"[Retrieve] Using filter expression: {expr}")
    else:
        logger.debug("[Retrieve] No filter expression applied")

    # Perform similarity search
    result = await db_manager.vector_store.asimilarity_search(
        question,
        k=4,
        ranker_type="rrf",
        ranker_params={"k": 60},
        expr=expr,
    )

    logger.info(f"[Retrieve] Retrieved {len(result)} documents for question.")
    return {"qc_pairs": [{"q": question, "ctx": result}]}


def _get_expression(state: RetrieveState):
    """
    Build vector store filter expression based on classifier.
    :param state: current graph state
    :return: filter expression string or None
    """
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None
