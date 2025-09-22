import asyncio
import logging
from openai import ContentFilterFinishReasonError
from pydantic import Field, BaseModel

from module_instances import model_manager
from modules.rag.prompts import generate_prompt, generate_multiple_prompt
from modules.rag.state import State, QC
from modules.rag.utils import invoke_model_and_receive_token_usage

logger = logging.getLogger(__name__)


class ResponseFormatter(BaseModel):
    """
    Structured output enforcing a plain text answer.
    :param answer: final answer text for the user
    """

    answer: str = Field(..., description="The answer to the user's question.")


async def generate(state: State):
    """
    Final answer generation step.
    - Single question: generate a direct answer
    - Multiple questions: fan-out into sub-answers, then fan-in into summary
    :param state: current graph state
    :return: dict with final answer and token statistics
    """
    qc_pairs = state["qc_pairs"]
    if not qc_pairs:
        logger.warning("[Generate] No questions found in state.")
        return {
            "answer": "Keine beantwortbare Frage vorhanden.",
            "input_tokens": 0,
            "output_tokens": 0,
            "token_usage": [],
        }

    # Single question → direct answer
    if len(qc_pairs) == 1:
        logger.info("[Generate] Processing single question.")
        return await generate_answer(qc_pairs[0])

    # Multiple questions → fan-out + fan-in
    logger.info(f"[Generate] Processing {len(qc_pairs)} questions (fan-out/fan-in).")

    # Fan-out: generate answers for each question
    answers = await asyncio.gather(*[generate_answer(qc) for qc in qc_pairs])

    # Prepare fan-in summary prompt
    formatted_pairs = "\n".join(
        f"- Q: {qc['q']}\n  A: {ans['answer']}" for qc, ans in zip(qc_pairs, answers)
    )
    user_input = state["user_input"]

    logger.debug(f"[Generate] Formatted pairs for summary:\n{formatted_pairs}")

    meta_messages = generate_multiple_prompt.invoke(
        {"user_input": user_input, "formatted_pairs": formatted_pairs}
    )

    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)
    final_response, summary_usage = invoke_model_and_receive_token_usage(
        structured_model, meta_messages, step="generate_summary"
    )

    logger.info("[Generate] Summary generation complete.")

    # Consolidate token statistics
    all_input_tokens = (
        sum(a["input_tokens"] for a in answers) + summary_usage["input_tokens"]
    )
    all_output_tokens = (
        sum(a["output_tokens"] for a in answers) + summary_usage["output_tokens"]
    )
    all_token_usage = sum((a["token_usage"] for a in answers), []) + [summary_usage]

    return {
        "answer": final_response.answer,
        "input_tokens": all_input_tokens,
        "output_tokens": all_output_tokens,
        "token_usage": all_token_usage,
    }


async def generate_answer(qc: QC):
    """
    Generate an answer for a single reformulated question using its context.
    :param qc: question-context pair
    :return: dict with answer and token statistics
    """
    docs_content = "\n\n".join(doc.page_content for doc in qc["ctx"])
    question = qc["q"]

    logger.info(f"[GenerateAnswer] Generating answer for: {question}")
    logger.debug(f"[GenerateAnswer] Context length: {len(docs_content)} chars")

    messages = generate_prompt.invoke({"question": question, "context": docs_content})
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    try:
        response, token_usage = invoke_model_and_receive_token_usage(
            structured_model, messages, step="generate_answer"
        )
        logger.info(f"[GenerateAnswer] Answer generated for question: {question}")
    except ContentFilterFinishReasonError as e:
        logger.warning(f"[GenerateAnswer] Content filter triggered: {e}")
        response = ResponseFormatter(answer="")
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    return {
        "answer": response.answer,
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "token_usage": [token_usage],
    }
