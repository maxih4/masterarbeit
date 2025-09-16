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
    """Structured output: erzwingt eine reine Textantwort."""
    answer: str = Field(..., description="The answer to the user's question.")


async def generate(state: State):
    """Finale Antwortgenerierung:
    - Einzelfrage: direkte Antwort
    - Mehrfachfragen: parallele Teilantworten (fan-out) + Zusammenfassung (fan-in)
    """
    qc_pairs = state["qc_pairs"]
    if not qc_pairs:
        return {"answer": "Keine beantwortbare Frage vorhanden.",
                "input_tokens": 0, "output_tokens": 0, "token_usage": []}

    # Einzelfrage → Short-circuit
    if len(qc_pairs) == 1:
        return await generate_answer(qc_pairs[0])

    logger.info("multiple questions")

    # Parallele Teilantworten (fan-out)
    answers = await asyncio.gather(*[generate_answer(qc) for qc in qc_pairs])

    # Zusammenfassungs-Prompt (fan-in)
    formatted_pairs = "\n".join(
        f"- Q: {qc['q']}\n  A: {ans['answer']}" for qc, ans in zip(qc_pairs, answers)
    )
    user_input = state["user_input"]
    meta_messages = generate_multiple_prompt.invoke(
        {"user_input": user_input, "formatted_pairs": formatted_pairs}
    )

    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)
    final_response, summary_usage = invoke_model_and_receive_token_usage(
        structured_model, meta_messages, step="generate_summary"
    )

    # Tokenstatistiken konsolidieren
    all_input_tokens = sum(a["input_tokens"] for a in answers) + summary_usage["input_tokens"]
    all_output_tokens = sum(a["output_tokens"] for a in answers) + summary_usage["output_tokens"]
    all_token_usage = sum((a["token_usage"] for a in answers), []) + [summary_usage]

    return {"answer": final_response.answer,
            "input_tokens": all_input_tokens,
            "output_tokens": all_output_tokens,
            "token_usage": all_token_usage}


async def generate_answer(qc: QC):
    """Beantwortet eine einzelne reformulierte Frage anhand ihres Kontextes."""
    docs_content = "\n\n".join(doc.page_content for doc in qc["ctx"])
    question = qc["q"]
    messages = generate_prompt.invoke({"question": question, "context": docs_content})
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    logger.info(messages)
    try:
        response, token_usage = invoke_model_and_receive_token_usage(
            structured_model, messages, step="generate_answer"
        )
    except ContentFilterFinishReasonError as e:
        logger.warning("Content filter triggered in generate_answer: %s", e)
        response = ResponseFormatter(answer="")
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    return {"answer": response.answer,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
            "token_usage": [token_usage]}
