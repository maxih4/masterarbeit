import logging
from modules.rag.state import State

logger = logging.getLogger(__name__)


def dont_know(state: State):
    """
    Node for unknown or irrelevant questions.
    :param state: current graph state
    :return: dict with default fallback answer
    """
    logger.info("[DontKnow] Reached fallback node (no suitable answer).")
    return {
        "answer": (
            "Tut mir leid, dabei kann ich Ihnen nicht helfen. "
            "Bitte versuchen Sie es mit einer anderen Frage erneut."
        )
    }
