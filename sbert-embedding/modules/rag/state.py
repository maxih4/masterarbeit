from typing import TypedDict, List, Dict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    question: str
    user_input: str
    context: List[Document]
    answer: str
    last_ai_message: str
    last_user_question: str
    classifier: str
    token_usage: Dict[str, Dict[str, int]]
