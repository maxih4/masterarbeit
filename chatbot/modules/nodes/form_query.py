# Form query node: reformulates user input into concrete questions

import logging
from typing import List, Literal

from langgraph.types import Send, Command
from openai import ContentFilterFinishReasonError
from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.nodes.retrieve import retrieve
from modules.rag.prompts import prompt_template
from modules.rag.state import State, RetrieveState
from modules.rag.utils import invoke_model_and_receive_token_usage

logger = logging.getLogger(__name__)


def form_query(state: State) -> Command[Literal["retrieve"]]:
    """
    Reformulate user input into one or more concrete questions.
    :param state: current graph state
    :return: Command that updates state and fans out to 'retrieve' node(s)
    """
    # Extract history
    last_questions = state.get("questions", [])
    last_answer = state.get("answer", "")

    questions_str = " | ".join(last_questions)

    logger.info("[FormQuery] Starting query formation.")
    logger.debug(f"[FormQuery] Last questions: {last_questions}")
    logger.debug(f"[FormQuery] Last answer: {last_answer}")

    # System instructions
    instructions = (
        "You are an expert assistant for a recycling company's question-answering system. "
        "Your task is to extract and rewrite multiple distinct and concrete user questions from a single input. "
        "Each question must: "
        "- Refer to a clearly different item or topic (no overlaps). "
        "- Avoid rephrasing the same meaning in different words. "
        "- Be clear, short, and self-contained."
    )

    # Examples
    positive_examples = (
        "Wohin gehört Metall entsorgt?",
        "Dürfen Dachziegel in den Sperrmüll?",
        "Darf in den Sperrmüll Container auch ein Fahrrad?",
        "Kann ich mit Paypal bezahlen?",
        "Wird ein Wunschtermin eingehalten?",
        "Wie lange im Vorraus muss ich einen Container bestellen?",
    )
    negative_examples = (
        "1. Können Holz, Bauschutt und Fliesen in den gleichen Container entsorgt werden? -> Multiple items combined. Wrong.",
        "2. Was gehört da rein? -> Not specific enough, missing entity.",
        "3. Gehört Glas in den Sperrmüll? Kann ich Glas in den Sperrmüll Container werfen? -> Duplicate meaning. Wrong.",
        "4. Können Metall und Glas im Sperrmüll abgelegt werden? -> Combined items. Wrong.",
    )

    # Combine user input + history into one string
    human_input_with_additional_information = (
        f"User Input: {state['user_input']}\n"
        f"Chat History: last_questions={questions_str}; last_answer={last_answer}"
    )

    # Structured output model
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    # Build prompt
    message = prompt_template.invoke(
        {
            "instructions": instructions,
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "human_input_with_additional_information": human_input_with_additional_information,
        }
    )

    logger.debug(f"[FormQuery] Prompt built for LLM: {message}")

    # Call model
    try:
        answer, token_usage = invoke_model_and_receive_token_usage(
            structured_model, message, "form_query"
        )
        questions = answer.questions
        logger.info(f"[FormQuery] Generated {len(questions)} questions.")
    except ContentFilterFinishReasonError as e:
        logger.warning(f"[FormQuery] Content filter triggered: {e}")
        questions = []
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    logger.debug(f"[FormQuery] Questions: {questions}")

    # Build sends (fan-out to retrieve)
    sends = [
        Send(
            "retrieve",
            RetrieveState(question=q, classifier=state["classifier"]),
        )
        for q in questions
    ]
    logger.debug(f"[FormQuery] Sends prepared: {sends}")

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
    """
    Structured output for query formation.
    :param questions: list of reformulated questions as plain strings
    """

    questions: List[str] = Field(
        ...,
        description=(
            "A list of reformulated or generated questions based on the user input. "
            "Each item must be a single, self-contained question."
        ),
    )
