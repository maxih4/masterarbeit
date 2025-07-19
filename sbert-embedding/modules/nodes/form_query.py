# Form query to one question


### TODO: if needed make multiple questions out of the user query
import logging
from typing import List

from pydantic import BaseModel, Field

from module_instances import model_manager
from modules.rag.prompts import form_query_prompt
from modules.rag.state import State

logger = logging.getLogger(__name__)


def form_query(state: State):
    # Get all questions from last time
    last_questions = state.get("questions", [])

    # Join with separator symbol
    questions_str = " | ".join(last_questions)

    # get the generated answer from last time
    last_answer = state.get("answer", "")
    logger.info(f"Last questions: {last_questions}")
    logger.info(f"Last answer: {last_answer}")

    # generate structured output model
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    # generate input message
    message = form_query_prompt.invoke(
        {
            "user_input": state["user_input"],
            "chat_history": "last_questions: "
            + questions_str
            + "last_answer: "
            + last_answer,
        },
    )

    # invoke model
    questions = structured_model.invoke(message)

    logger.info(f"Generated Questions: {questions}")

    return {"questions": questions}


class ResponseFormatter(BaseModel):
    questions: List[str] = Field(
        ...,
        description=(
            "A list of reformulated or generated questions based on the user input. "
            "Each item in the list is a single question as a string."
        ),
    )
