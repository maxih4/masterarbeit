# Define steps
from module_instances import create_db_manager
from modules.rag.state import State


def retrieve(state: State):
    db_manager = create_db_manager(drop_old=False)
    retrieved_docs = db_manager.vector_store.similarity_search(
        state["question"],
        k=3,
        ranker_type="rrf",
        ranker_params={"k": 100},
        expr=_get_expression(state),
    )

    retrieved_docs_weighted = db_manager.vector_store.similarity_search(
        state["question"],
        k=3,
        ranker_type="weighted",
        ranker_params={"weights": [0.9, 0.1]},
        expr=_get_expression(state),
    )
    for doc in retrieved_docs:
        print("Doc metadata:", doc.metadata)

    print("Weighted Docs:")
    for doc in retrieved_docs_weighted:
        print("Weighted Doc metadata:", doc.metadata)
    return {"context": retrieved_docs}


def _get_expression(state: State):
    if state["classifier"] == "internal_faq":
        return 'source == "csv/faq.csv"'
    if state["classifier"] == "waste_disposal_guidance":
        return 'source == "csv/fraktionen.csv"'
    return None
