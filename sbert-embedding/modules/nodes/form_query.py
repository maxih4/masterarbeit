# Form query to one question
from langchain_core.messages import AIMessage

from module_instances import model_manager
from modules.rag.prompts import first_prompt
from modules.rag.state import State


def form_query(self, state: State):
        last_question = state.get("last_user_question", [])
        question_prompt = first_prompt.invoke({
            "user_input": state["user_input"],
            "chat_history": last_question
        })

        # structured_llm = model_manager.llm_model.with_structured_output(ClassifiedOutput)
        # question = structured_llm.invoke(question_prompt)
        question = model_manager.llm_model.invoke(question_prompt)
        new_ai_message = AIMessage(content=question.content)

        return {"question": question.content, "last_user_question": [question.content]}
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