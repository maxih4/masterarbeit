from operator import add
from typing import TypedDict, List, Dict, Annotated

from langchain_core.documents import Document


class QA(TypedDict):
    q: str
    ctx: List[Document]


class State(TypedDict):
    questions: List[str]
    user_input: str
    answer: str
    last_answer: str
    last_user_questions: List[str]
    classifier: str
    token_usage: Dict[str, Dict[str, int]]
    qa_pairs: Annotated[list[QA], add]


class RetrieveState(TypedDict):
    classifier: str
    question: str
