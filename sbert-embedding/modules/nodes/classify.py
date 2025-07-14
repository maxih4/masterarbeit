from typing import Literal

from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.rag.prompts import classify_prompt
from modules.rag.state import State
from modules.rag.utils import get_token_usage

# TODO: Add one more classification for complex inquiries e.g. orders, or complaints


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
        "internal_faq",
        "waste_disposal_guidance",
        "irrelevant_or_smalltalk",
        "complex_query_customer_support",
    ] = Field(
        ...,
        description=(
            "The classifier category for the user input. "
            "One of: "
            "'internal_faq' — general information about the company, internal processes, payments, invoices, contact details, etc. "
            "'waste_disposal_guidance' — how to dispose of specific materials, what goes into which bin/container, recycling instructions, etc. "
            "'irrelevant_or_smalltalk' — off-topic questions, chit-chat, jokes, greetings, or anything unrelated to the recycling company. "
            "'complex_query_customer_support' — specific or complex questions about individual orders, deliveries, complaints, or service issues "
            "(e.g., 'When will my container be delivered?', 'Why wasn’t my container emptied?', 'My delivery is late', 'I want to file a complaint')."
        ),
    )


def classify_path_function(
    state: State,
) -> Literal["dont_know", "form_query", "contact_customer_support"]:
    if state["classifier"] == "irrelevant_or_smalltalk":
        return "dont_know"
    if state["classifier"] == "complex_query_customer_support":
        return "contact_customer_support"
    else:
        return "form_query"
