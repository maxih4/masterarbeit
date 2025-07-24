import asyncio

from pydantic import Field, BaseModel

from module_instances import model_manager
from modules.rag.prompts import generate_prompt, generate_multiple_prompt
from modules.rag.state import State, QA
from modules.rag.utils import invoke_model_and_receive_token_usage


async def generate(state: State):
    qa_pairs = state["qa_pairs"]

    # Shortcut for single question
    if len(qa_pairs) == 1:
        return await generate_answer(qa_pairs[0], "generate_answer")

    print("multiple questions")

    # Generate answers in parallel
    answers = await asyncio.gather(
        *[generate_answer(qa, "generate_answer") for qa in qa_pairs]
    )

    # Prepare input for summarization
    formatted_pairs = "\n".join(
        f"- Q: {qa['q']}\n  A: {ans['answer']}" for qa, ans in zip(qa_pairs, answers)
    )
    user_input = state["user_input"]

    # Final summarization call
    meta_messages = generate_multiple_prompt.invoke(
        {"user_input": user_input, "formatted_pairs": formatted_pairs}
    )

    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)
    final_response, summary_usage = invoke_model_and_receive_token_usage(
        structured_model, meta_messages, step="generate_summary"
    )

    # Merge usage stats
    all_input_tokens = (
        sum(ans["input_tokens"] for ans in answers) + summary_usage["input_tokens"]
    )
    all_output_tokens = (
        sum(ans["output_tokens"] for ans in answers) + summary_usage["output_tokens"]
    )
    all_token_usage = sum((ans["token_usage"] for ans in answers), []) + [summary_usage]

    return {
        "answer": final_response.answer,
        "input_tokens": all_input_tokens,
        "output_tokens": all_output_tokens,
        "token_usage": all_token_usage,
    }


async def generate_answer(qa: QA, state: State):
    docs_content = "\n\n".join(doc.page_content for doc in qa["ctx"])
    question = qa["q"]
    messages = generate_prompt.invoke({"question": question, "context": docs_content})
    structured_model = model_manager.llm_model.with_structured_output(ResponseFormatter)

    # Use your utility
    response, token_usage = invoke_model_and_receive_token_usage(
        structured_model,
        messages,
        step="generate_answer",
    )

    return {
        "answer": response.answer,
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "token_usage": [token_usage],
    }


class ResponseFormatter(BaseModel):
    answer: str = Field(
        ...,
        description="The answer to the users question.",
    )
