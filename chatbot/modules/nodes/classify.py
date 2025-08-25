from typing import Literal

from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.rag.prompts import prompt_template
from modules.rag.state import State
from modules.rag.utils import invoke_model_and_receive_token_usage


def classify(state: State):
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    instructions = (
        "You are an intent-classifier for a question-answering system."
        "Your job is to classify the user input into exactly one of these four categories:"
        "- 'internal_faq':"
        "The user asks for general information about the company, internal processes, payments, invoices, contact details, general questions regarding the process of ordering or FAQ topics."
        "- 'waste_disposal_guidance':"
        "The user asks for general information about how to correctly dispose of a material, what goes into which bin/container, or what type of container is appropriate for specific waste."
        "- 'irrelevant_or_smalltalk':"
        "Off-topic questions, chit-chat, jokes, greetings, or anything unrelated to the recycling company."
        "- 'complex_query_customer_support':"
        "The user requests a specific service action, asks for a price of a container or has a complaint."
        "Only choose one of these four categories. Do not explain your choice."
        "Always focus on the *intent of the user*:"
        "- If the user only asks for information or advice (e.g., how-to, what type), choose `internal_faq`."
        "- If the user asks for an action to be performed (e.g., pickup, delivery, order, complaint, price), choose `complex_query_customer_support`."
        "- If the user asks where to put anything (e.g. in what container), choose `waste_disposal_guidance`."
    )

    positive_examples = (
        "Wie entsorge ich eine alte Matratze? Classification: waste_disposal_guidance"
        "Steinboden, Dachpappe, Regenrinnen. Wo entsorge ich das? Classification: waste_disposal_guidance"
        "Kann ich per Rechnung bezahlen? Classification: internal_faq"
        "Kann ich den Container drehen? Classification: internal_faq"
        "Hallo, wie geht es dir? Classification: irrelevant_or_smalltalk"
        "Wie spät ist es? Classification: irrelevant_or_smalltalk"
        "Können Sie die Papiertonne in der Straße 123 abholen? Die ABholung ist heute irgendwie ausgefallen. Classification: complex_query_customer_support"
        "Machen Sie mir ein Angebot für Speisereste. Ich habe circa 5m³. Classification: complex_query_customer_support",
    )

    negative_examples = (
        # Wrong category: customer support issue misclassified as FAQ
        "Meine Container wurden nicht geleert. Classification: internal_faq -> Falsch. Das ist keine allgemeine Frage, sondern ein konkretes Anliegen. "
        "Korrekte Classification: complex_query_customer_support",
        # Unclassified input
        "Darf Bauschutt in den Sperrmüll Container? -> Es fehlt die Classification. Korrekte Classification: waste_disposal_guidance",
        # Wrong category: joke misclassified as disposal question
        "Was passiert, wenn ich meinen Ex in die Tonne werfe? Classification: waste_disposal_guidance -> Falsch. Off-topic, scherzhaft. "
        "Korrekte Classification: irrelevant_or_smalltalk",
        # Wrong category: payment method misclassified as disposal
        "Wie kann ich bezahlen? Classification: waste_disposal_guidance -> Falsch. Geht um Zahlungsmodalitäten. "
        "Korrekte Classification: internal_faq",
        # Wrong category: small talk misclassified as support
        "Na, alles klar bei dir? Classification: complex_query_customer_support -> Falsch. Das ist Smalltalk. "
        "Korrekte Classification: irrelevant_or_smalltalk",
        # Wrong category: vague complaint misclassified as waste guidance
        "Warum ist mein Müll heute noch nicht abgeholt? Classification: waste_disposal_guidance -> Falsch. Es geht um ein konkretes Problem. "
        "Korrekte Classification: complex_query_customer_support"
        # Wrong category: multipel sentences misleading to complexity, but its a normal internal_faq
        "Heute ist unser Container schon voll und ich möchte wissen,wann die Container entleert werden. Classification: complex_query_customer_support -> Falsch. Es ist eine konkrete Frage, die in der FAQ stehen könnte."
        "Korrekte Classification: internal_faq",
    )

    human_input_with_additional_information = f"User input: {state["user_input"]}"

    messages = prompt_template.invoke(
        {
            "instructions": instructions,
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "human_input_with_additional_information": human_input_with_additional_information,
        }
    )

    answer, token_usage = invoke_model_and_receive_token_usage(
        structured_model, messages, "classify"
    )
    return {
        "classifier": answer.classifier,
        "token_usage": [token_usage],
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
    }


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
