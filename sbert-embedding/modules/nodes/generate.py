from module_instances import model_manager
from modules.rag.prompts import generate_prompt
from modules.rag.state import State
from modules.rag.utils import get_token_usage


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    # get all questions
    questions = state["questions"]

    # conc them
    questions_str = " | ".join(questions)

    messages = generate_prompt.invoke(
        {"questions": questions_str, "context": docs_content}
    )
    response = model_manager.llm_model.invoke(messages)
    tokens = get_token_usage("generate", messages.to_string(), response.content, state)

    return {"answer": response.content, **tokens}
