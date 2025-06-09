from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain.prompts import ChatPromptTemplate
from psycopg_pool import AsyncConnectionPool
from modules import  db_manager,model_manager
from langchain.schema import HumanMessage, AIMessage
from typing import Literal, TypedDict, Optional, List, Union
from langchain.schema.document import Document
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field
from langgraph.types import Command







first_prompt = ChatPromptTemplate([
    ("system", 
     "You are a multilingual assistant that prepares user inputs for a company-specific question-answering system. "
     "Follow these steps carefully:\n\n"
     "1. Determine whether the user input is:\n"
     "   a) A company-specific request (e.g. services, pricing, recycling procedures, containers, scheduling, etc.)\n"
     "   b) Small talk or general chit-chat (e.g. 'How are you?', jokes, personal questions)\n\n"
     "2. If the input is small talk or irrelevant, do NOT provide an answer. Instead, respond politely but clearly, in the user's language, with something like:\n"
     "   'Dafür habe ich leider keine Zeit – ich helfe gerade anderen Kunden. Hast du eine Frage zu unseren Dienstleistungen?'\n"
     "   or\n"
     "   'I don't have time for that – I'm currently helping other customers. Do you have a question about our services?'\n\n"
     "3. If the input is company-specific:\n"
     "   - Reformulate it into a concise, clear question as if coming directly from the user.\n"
     "   - Use the latest chat history if relevant to disambiguate or complete the question.\n"
     "   - If the question is already well-formed, repeat it unchanged.\n\n"
     "Important:\n"
     "- Never answer factual questions yourself.\n"
     "- Only provide one question at maximum"
     "- Do not change the perspective of the question (e.g. if the user asks 'What are your services?', do not change it to 'What services do I offer?').\n"
     "- Always preserve and reflect the user's language (e.g. German input → German output).\n"
     "- Output **either** a reformulated question **or** the polite refusal message. Never both."
    ),
    ("human", 
     "User Input: {user_input}\n\n"
     "Last Message in Chat History (optional): {chat_history}"
    )
])

second_prompt = ChatPromptTemplate([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the question."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

class ClassifiedOutput(BaseModel):
    type: Literal["question", "polite_answer"] = Field(description="Specifies whether the content is a question or a polite answer to small talk")
    content: str = Field(description="Either the reformulated question or the direct answer")

# Define state for application
class State(TypedDict):
    question: str
    user_input: str
    context: List[Document]
    answer: str
    last_user_message: List[HumanMessage]
    last_ai_message: List[AIMessage]
    last_user_question: List[AIMessage]




class RagManager:
    def __init__(self):
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None





    async def create_graph(self) -> Optional[CompiledStateGraph]:
        if self._graph is None:
            try:
                graph_builder = StateGraph(State)
                graph_builder.add_edge(START, "_form_query",)
                graph_builder.add_node("_retrieve", self._retrieve)
                graph_builder.add_node("_form_query", self._form_query)
                graph_builder.add_node("_generate", self._generate)
                graph_builder.add_edge("_retrieve", "_generate")
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool) # type: ignore
                    await checkpointer.setup()
                self._graph =  graph_builder.compile(checkpointer=checkpointer)
            except Exception as e:
                raise e
        return self._graph

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        if self._connection_pool is None:
            try:
                self._connection_pool = AsyncConnectionPool(
                    "postgresql://postgres:example@localhost:5432/postgres?sslmode=disable",
                    open=False,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
            except Exception as e:
                raise e
        return self._connection_pool
    

    # Form query to one question
    def _form_query(self, state: State):
        last_question = state.get("last_user_question", [])
        question_prompt = first_prompt.invoke({
            "user_input": state["user_input"],
            "chat_history": last_question
        })
        
        structured_llm = model_manager.llm_model.with_structured_output(ClassifiedOutput)
        question = structured_llm.invoke(question_prompt)
       # question = model_manager.llm_model.invoke(question_prompt)
        #new_ai_message = AIMessage(content=question.content)
       # return {"question": question.content, "last_user_question": [new_ai_message]}
        if isinstance(question, ClassifiedOutput):
            if (question.type=="question"):
                goto = "_retrieve"
            if( question.type=="polite_answer"):
                goto = "END"
            # polite_answer = question.content
            # new_ai_message = AIMessage(content=polite_answer)
            # state["last_ai_message"] = [new_ai_message]
            # state["answer"] = polite_answer
            # return {"question": "", "last_user_question": [], "last_ai_message": [new_ai_message], "answer": polite_answer}
            return Command(update={
            "question": question.content,
            "last_user_question": question.content,
            }, goto=goto)

    # Define steps
    def _retrieve(self, state: State):
        retrieved_docs = db_manager.vector_store.similarity_search(state["question"], k=3, ranker_type="rrf", ranker_params={"k": 100})
        return {"context": retrieved_docs}


    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = second_prompt.invoke({"question": state["question"], "context": docs_content})
        response = model_manager.llm_model.invoke(messages)
        return {"answer": response.content}




######https://github.com/langchain-ai/langgraph/discussions/894#discussioncomment-10277417
