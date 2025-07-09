from typing import TypedDict, List, Dict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    question: str
    user_input: str
    context: List[Document]
    answer: str
    last_user_message: List[HumanMessage]
    last_ai_message: List[AIMessage]
    last_user_question: List[AIMessage]
    classifier: str
    token_usage: Dict[str, Dict[str, int]]
