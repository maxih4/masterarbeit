from module_instances import anonymizer_manager
from modules.rag.state import State


def anonymize(state: State):
    unformatted = state["user_input"]
    formatted = anonymizer_manager.anonymizer.anonymize(unformatted)
    return {"user_input": formatted}
