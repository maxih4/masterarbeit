# Form query to one question


import logging
from typing import List, Literal

from langgraph.types import Send, Command
from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.nodes.retrieve import retrieve
from modules.rag.prompts import prompt_template
from modules.rag.state import State, RetrieveState
from modules.rag.utils import invoke_model_and_receive_token_usage

logger = logging.getLogger(__name__)


def form_query(state: State) -> Command[Literal["retrieve"]]:
    # Get all questions from last time
    last_questions = state.get("questions", [])

    # Join with separator symbol
    questions_str = " | ".join(last_questions)

    # get the generated answer from last time
    last_answer = state.get("answer", "")
    logger.info(f"Last questions: {last_questions}")
    logger.info(f"Last answer: {last_answer}")

    instructions = (
        "You are an expert assistant for a recycling company's question-answering system. "
        "Your task is to extract and rewrite multiple distinct and concrete user questions from a single input. "
        "Each question must: "
        "- Refer to a clearly different item or topic (no overlaps). "
        "- Avoid rephrasing the same meaning in different words. "
        "- Be clear, short, and self-contained."
    )

    positive_examples = (
        "Wohin gehört Metall entsorgt?",
        "Dürfen Dachziegel in den Sperrmüll?",
        "Darf in den Sperrmüll Container auch ein Fahrrad?"
        "Kann ich mit Paypal bezahlen?"
        "Wird ein Wunschtermin eingehalten?",
        "Wie lange im Vorraus muss ich einen Container bestellen?",
    )

    negative_examples = (
        "1. Können Holz, Bauschutt und Fliesen in den gleichen Container entsorgt werden? -> Mehrere Dinge zusammengeführt zu einer Frage. Nur eine Entitität pro Frage",
        "2. Was gehört da rein? -> Nicht spezifisch genug, fehlende Entititäten",
        "3. Gehört Glas in den Sperrmüll? Kann ich Glas in den Sperrmüll Container werfen? -> Mehrfach die Gleiche Frage. Nur eine Frage zum gleichen Thema",
        "4. 'Können Metall und Glas im Sperrmüll abgelegt werden? -> Mehrere Abfallfragen kombiniert. FALSCH!  ",
    )
    human_input_with_additional_information = (
        f"User Input: {state["user_input"]}",
        f"Chat History (optional): last_questions: {questions_str} last_answer: {last_answer}",
    )

    # generate structured output model
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    # generate input message

    message = prompt_template.invoke(
        {
            "instructions": instructions,
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "human_input_with_additional_information": human_input_with_additional_information,
        }
    )

    # invoke model
    answer, token_usage = invoke_model_and_receive_token_usage(
        structured_model, message, "form_query"
    )
    questions = answer.questions

    logger.info(f"Generated Questions: {questions}")
    sends = [
        Send(
            "retrieve",
            RetrieveState(question=q, classifier=state["classifier"]),
        )
        for q in questions
    ]

    # `update` is the normal state‑update dict,
    # `sends` is the fan‑out payload.
    return Command(
        update={
            "questions": questions,
            "token_usage": [token_usage],
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        },
        goto=sends,
    )


class ResponseFormatter(BaseModel):
    questions: List[str] = Field(
        ...,
        description=(
            "A list of reformulated or generated questions based on the user input. "
            "Each item in the list is a single question as a string."
        ),
    )
