from typing import TypedDict, List, Dict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    questions: List[str]
    user_input: str
    context: List[Document]
    answer: str
    last_answer: str
    last_user_questions: List[str]
    classifier: str
    token_usage: Dict[str, Dict[str, int]]
