from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langchain.prompts import ChatPromptTemplate
from modules import  db_manager,model_manager


prompt: ChatPromptTemplate = ChatPromptTemplate([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. You are only allowed to answer using the same language as the question."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define steps
def retrieve(state: State):
    retrieved_docs = db_manager.vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model_manager.llm_model.invoke(messages)
    return {"answer": response.content}


# Create the graph (compile once, reuse)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()