import logging
from module_instances import anonymizer_manager
from modules.rag.state import State

logger = logging.getLogger(__name__)


def anonymize(state: State):
    """
    First node in the graph: anonymize user input.
    :param state: current graph state
    :return: dict with anonymized user input
    """
    unformatted = state["user_input"]
    logger.info("[Anonymize] Starting anonymization of user input.")
    logger.debug(f"[Anonymize] Original input: {unformatted!r}")

    formatted = anonymizer_manager.anonymizer.anonymize(unformatted)

    logger.debug(f"[Anonymize] Anonymized input: {formatted!r}")
    logger.info("[Anonymize] Anonymization complete.")

    return {"user_input": formatted}
