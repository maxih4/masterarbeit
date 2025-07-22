from module_instances import model_manager
from modules.rag.prompts import generate_prompt
from modules.rag.state import State, QA
from modules.rag.utils import get_token_usage


async def generate(state: State):

    #  Wenn eine frage, dann direkt generieren
    if len(state["qa_pairs"]) == 1:
        return await generate_answer(qa=state["qa_pairs"][0], state=state)

    else:
        print("multiple questions")
        return {}

        # answers = await asyncio.gather(
        #     *[generate_answer(qa) for qa in state["qa_pairs"]]
        # )
        #
        # # Schritt 2: Meta-Antwort mit user_input + vorherigen Antworten
        # combined_context = "\n\n".join(
        #     f"Frage: {entry['question']}\nAntwort: {entry['answer']}"
        #     for entry in answers
        # )
        # user_input = state["user_input"]
        # meta_messages = generate_prompt.invoke(
        #     {"question": user_input, "context": combined_context}
        # )
        # final_response = await model_manager.llm_model.ainvoke(meta_messages)
        #
        # # Schritt 3: optionales Token-Tracking
        # total_tokens = get_token_usage(
        #     "generate-multi",
        #     meta_messages.to_string(),
        #     final_response.content,
        #     state,
        # )
        #
        # return {
        #     "intermediate_answers": answers,
        #     "answer": final_response.content,
        #     **total_tokens,
        # }

        # return {"answer": response.content, **tokens}


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
