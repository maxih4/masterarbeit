# Define steps
import asyncio

from module_instances import create_db_manager
from modules.rag.state import State


async def retrieve(state: State):

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

    questions = state["questions"]
    # for each qeustion form an asimilarity-search

    tasks = [retrieve_documents_for_question(question, state) for question in questions]
    # await all and pass to llm
    all_results = await asyncio.gather(*tasks)

    # flatten
    flat_docs = [doc for sublist in all_results for doc in sublist]

    # deduplicate by metadata key
    seen = set()
    unique_docs = []

    for doc in flat_docs:
        key = doc.metadata.get("pk")
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return {"context": unique_docs}


def _get_expression(state: State):
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None


async def retrieve_documents_for_question(question: str, state: State):
    db_manager = create_db_manager(drop_old=False)
    return await db_manager.vector_store.asimilarity_search(
        question,
        k=3,
        ranker_type="rrf",  # or "weighted"
        ranker_params={"k": 100},  # or {"weights": [0.9, 0.1]} if using "weighted"
        expr=_get_expression(state),
    )
