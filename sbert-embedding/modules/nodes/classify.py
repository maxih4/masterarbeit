from typing import Literal

from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.rag.prompts import classify_prompt
from modules.rag.state import State
from modules.rag.utils import get_token_usage


def classify(state: State):
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)
    messages = classify_prompt.invoke({"user_input": state["user_input"]})
    print(messages)
    answer = structured_model.invoke(messages)
    print(answer)
    tokens = get_token_usage("classify", messages.to_string(), answer.classifier, state)
    print(tokens)
    return {"classifier": answer.classifier, **tokens}


class ResponseFormatter(BaseModel):
    classifier: Literal[
        "internal_faq", "waste_disposal_guidance", "irrelevant_or_smalltalk"
    ] = Field(
        ...,
        description=(
            "The classifier category for the user input. "
            "One of: "
            "'internal_faq' (questions about orders, payments, or internal company processes), "
            "'waste_disposal_guidance' (questions about what goes into which container or how to dispose of something), "
            "'irrelevant_or_smalltalk' (chit-chat, jokes, or off-topic, nothing to do with a recycling company)."
        ),
    )


def classify_path_function(state: State) -> Literal["dont_know", "form_query"]:
    if state["classifier"] == "irrelevant_or_smalltalk":
        return "dont_know"
    else:
        return "form_query"
