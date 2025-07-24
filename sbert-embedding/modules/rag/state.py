from operator import add
from typing import TypedDict, List, Dict, Annotated

from langchain_core.documents import Document


# custom class for question to context mapping
class QA(TypedDict):
    q: str
    ctx: List[Document]


# Custom state for multiple parallel retrieve nodes
class RetrieveState(TypedDict):
    classifier: str
    question: str


# custom class for token usage
class TokenUsageEntry(TypedDict):
    step_name: str
    input_tokens: int
    output_tokens: int


class State(TypedDict):
    questions: List[str]
    user_input: str
    answer: str
    last_answer: str
    last_user_questions: List[str]
    classifier: str
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    token_usage: Annotated[List[TokenUsageEntry], add]
    qa_pairs: Annotated[list[QA], add]
