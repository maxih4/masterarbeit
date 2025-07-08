import logging
from langgraph.graph import START, StateGraph
from langchain.prompts import ChatPromptTemplate
from psycopg_pool import AsyncConnectionPool
from module_instances import  db_manager,model_manager
from langchain.schema import HumanMessage, AIMessage
from typing import Literal, TypedDict, Optional, List, Union
from langchain.schema.document import Document
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver






first_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert assistant for a recycling company's question-answering system. "
        "Your task is to rephrase the user's input into a clear, semantically meaningful, and complete question. "
        "Preserve exactly the meaning and scope of the original input — do not add or assume any information that is not explicitly present. "
        "Use the chat history only to resolve ambiguities, not to add unrelated topics. "
        "Output only the improved question, nothing else."
        "Do not add any words that are not needed to form a question. "
    ),
    (
        "human",
        "User Input: {user_input}\n\n"
        "Chat History (optional): {chat_history}"
    )
])

second_prompt = ChatPromptTemplate([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Don't use more sentences then you need. You are only allowed to answer using the same language as the question."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])


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
                graph_builder.add_node("_retrieve", self._retrieve)
                graph_builder.add_node("_form_query", self._form_query)
                graph_builder.add_node("_generate", self._generate)
                graph_builder.add_edge(START, "_form_query")
                graph_builder.add_edge("_form_query", "_retrieve")
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
        
        #structured_llm = model_manager.llm_model.with_structured_output(ClassifiedOutput)
        #question = structured_llm.invoke(question_prompt)
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


    # Define steps
    def _retrieve(self, state: State):
        retrieved_docs = db_manager.vector_store.similarity_search(state["question"], k=3, ranker_type="rrf", ranker_params={"k": 100})
        for doc in retrieved_docs:
            print("Doc metadata:", doc.metadata)
        return {"context": retrieved_docs}


    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = second_prompt.invoke({"question": state["question"], "context": docs_content})
        response = model_manager.llm_model.invoke(messages)
        return {"answer": response.content}




######https://github.com/langchain-ai/langgraph/discussions/894#discussioncomment-10277417
