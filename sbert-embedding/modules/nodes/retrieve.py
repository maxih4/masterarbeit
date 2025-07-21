# Define steps
import asyncio

from module_instances import create_db_manager
from modules.rag.state import RetrieveState


async def retrieve(state: RetrieveState):

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

    print(state)
    result = await db_manager.vector_store.asimilarity_search(
        question,
        k=3,
        ranker_type="rrf",  # or "weighted"
        ranker_params={"k": 100},  # or {"weights": [0.9, 0.1]} if using "weighted"
        expr=_get_expression(state),
    )

    return {"qa_pairs": {"q": question, "ctx": result}}


async def _get_expression(state: RetrieveState):
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None
