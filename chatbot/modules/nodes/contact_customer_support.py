import logging
from modules.rag.state import State

logger = logging.getLogger(__name__)


def contact_customer_support(state: State):
    """
    Node for complex inquiries or complaints requiring human support.
    :param state: current graph state
    :return: dict with message directing user to customer service
    """
    logger.info("[ContactSupport] User should contact our customer support.")
    return {"answer": "Bitte kontaktieren Sie unseren Kundenservice."}
