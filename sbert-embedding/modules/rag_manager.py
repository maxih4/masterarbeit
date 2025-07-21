from langgraph.graph import START, StateGraph
from psycopg_pool import AsyncConnectionPool
from typing import Literal, Optional
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from modules.nodes.classify import classify, classify_path_function
from modules.nodes.contact_customer_support import contact_customer_support
from modules.nodes.dont_know import dont_know
from modules.nodes.form_query import form_query
from modules.nodes.generate import generate
from modules.nodes.retrieve import retrieve
from modules.rag.state import State


# Define state for application


class RagManager:
    def __init__(self):
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        if self._graph is None:
            try:
                graph_builder = StateGraph(State)
                graph_builder.add_node("retrieve", retrieve)
                graph_builder.add_node("form_query", form_query)
                graph_builder.add_node("generate", generate)
                graph_builder.add_node("classify", classify)
                graph_builder.add_node("dont_know", dont_know)
                graph_builder.add_node(
                    "contact_customer_support", contact_customer_support
                )
                graph_builder.add_edge(START, "classify")
                graph_builder.add_conditional_edges(
                    "classify", path=classify_path_function
                )
                graph_builder.add_edge("retrieve", "generate")
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)  # type: ignore
                    await checkpointer.setup()
                self._graph = graph_builder.compile(checkpointer=checkpointer)
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


######https://github.com/langchain-ai/langgraph/discussions/894#discussioncomment-10277417
