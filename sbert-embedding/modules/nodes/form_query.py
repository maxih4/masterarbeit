# Form query to one question


### TODO: if needed make multiple questions out of the user query
import logging
from module_instances import model_manager
from modules.rag.prompts import first_prompt
from modules.rag.state import State

logger = logging.getLogger(__name__)


def form_query(state: State):
    last_question = state.get("question", [])
    last_answer = state.get("answer", [])
    logger.info(f"Last question: {last_question}")
    logger.info(f"Last answer: {last_answer}")
    question_prompt = first_prompt.invoke(
        {
            "user_input": state["user_input"],
            "chat_history": "last_question: "
            + last_question
            + "last_answer: "
            + last_answer,
        },
    )

    question = model_manager.llm_model.invoke(question_prompt)
    logger.info(f"Generated Question: {question}")

    return {"question": question.content}
    # if isinstance(question, ClassifiedOutput):
    #     if (question.type=="question"):
    #         goto = "_retrieve"
    #     if( question.type=="polite_answer"):
    #         goto = "END"
    #     # polite_answer = question.content
    #     # new_ai_message = AIMessage(content=polite_answer)
    #     # state["last_ai_message"] = [new_ai_message]
    #     # state["answer"] = polite_answer
    #     # return {"question": "", "last_user_question": [], "last_ai_message": [new_ai_message], "answer": polite_answer}
    #     return Command(update={
    #     "question": question.content,
    #     "last_user_question": question.content,
    #     }, goto=goto)
