from typing import TypedDict, List, Annotated

from langchain_core.documents import Document


# Custom state for multiple parallel retrieve nodes
class RetrieveState(TypedDict):
    """
    State used for parallel retrieve nodes.
    :param classifier: classification label
    :param question: user question
    """

    classifier: str
    question: str


def merge_or_reset(old: list, new: list):
    """
    Merge or reset a list.
    :param old: current list
    :param new: new list, resets if falsy
    :return: merged or reset list
    """
    if not new:  # [] or None => reset
        return []
    return old + new  # merge


def add_or_reset(old: int, new: int):
    """
    Add or reset a counter.
    :param old: current counter
    :param new: new value, resets if falsy
    :return: added or reset value
    """
    if not new:  # 0 or None => reset
        return 0
    return old + new  # add


class QC(TypedDict):
    """
    Mapping question to context documents.
    :param q: user question
    :param ctx: related context documents
    """

    q: str
    ctx: List["Document"]


class TokenUsageEntry(TypedDict):
    """
    Token usage entry for one step.
    :param step_name: name of the pipeline step
    :param input_tokens: tokens used as input
    :param output_tokens: tokens used as output
    """

    step_name: str
    input_tokens: int
    output_tokens: int


class State(TypedDict):
    """
    State object for chatbot/graph execution.
    :param questions: generated questions
    :param user_input: last user input
    :param answer: current answer
    :param last_answer: previous answer
    :param last_user_questions: recent user questions
    :param classifier: classification
    :param input_tokens: tracked input tokens (add/reset)
    :param output_tokens: tracked output tokens (add/reset)
    :param token_usage: list of token usage entries per step(merge/reset)
    :param qc_pairs: list of question-context mappings (merge/reset)
    """

    questions: List[str]
    user_input: str
    answer: str
    last_answer: str
    last_user_questions: List[str]
    classifier: str
    input_tokens: Annotated[int, add_or_reset]
    output_tokens: Annotated[int, add_or_reset]
    token_usage: Annotated[List[TokenUsageEntry], merge_or_reset]
    qc_pairs: Annotated[List[QC], merge_or_reset]
