from module_instances import model_manager
from modules.rag.prompts import second_prompt
from modules.rag.state import State
from modules.rag.utils import get_token_usage


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = second_prompt.invoke(
        {"question": state["questions"][0], "context": docs_content}
    )
    response = model_manager.llm_model.invoke(messages)
    tokens = get_token_usage("generate", messages.to_string(), response.content, state)

    return {"answer": response.content, **tokens}
