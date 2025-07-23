import asyncio

from module_instances import model_manager
from modules.rag.prompts import generate_prompt, generate_multiple_prompt
from modules.rag.state import State, QA
from modules.rag.utils import get_token_usage


async def generate(state: State):

    #  Wenn eine frage, dann direkt generieren
    if len(state["qa_pairs"]) == 1:
        return await generate_answer(qa=state["qa_pairs"][0], state=state)

    else:
        print("multiple questions")

        answers = await asyncio.gather(
            *[generate_answer(qa, state) for qa in state["qa_pairs"]]
        )

        qa_with_answers = [
            {"q": qa["q"], "a": ans["answer"]}
            for qa, ans in zip(state["qa_pairs"], answers)
        ]

        user_input = state["user_input"]
        formatted_pairs = "\n".join(
            f"- Q: {pair['q']}\n  A: {pair['a']}" for pair in qa_with_answers
        )

        meta_messages = generate_multiple_prompt.invoke(
            {"user_input": user_input, "formatted_pairs": formatted_pairs}
        )
        final_response = await model_manager.llm_model.ainvoke(meta_messages)

        # Schritt 3: optionales Token-Tracking
        tokens = get_token_usage(
            "generate-multi",
            meta_messages.to_string(),
            final_response.content,
            state,
        )

        return {"answer": final_response.content, **tokens}


async def generate_answer(qa: QA, state: State):
    docs_content = "\n\n".join(doc.page_content for doc in qa["ctx"])
    question = qa["q"]
    messages = generate_prompt.invoke({"question": question, "context": docs_content})
    response = await model_manager.llm_model.ainvoke(messages)
    tokens = get_token_usage("generate", messages.to_string(), response.content, state)
    return {
        "answer": response.content,
        **tokens,
    }
