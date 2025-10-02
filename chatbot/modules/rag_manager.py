import os

from langgraph.graph import START, StateGraph
from psycopg_pool import AsyncConnectionPool
from typing import Optional
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from module_instances import create_db_manager
from modules.nodes import anonymize
from modules.nodes.classify import classify, classify_path_function
from modules.nodes.contact_customer_support import contact_customer_support
from modules.nodes.dont_know import dont_know
from modules.nodes.form_query import form_query
from modules.nodes.generate import generate
from modules.nodes.retrieve import retrieve
from modules.rag.state import State

from modules.nodes.anonymize import anonymize


class RagManager:
    def __init__(self):
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

        """
        Initialize the RagManager with a connection pool and a graph
        """

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        if self._graph is None:
            try:
                # Add all nodes and edges to the graph
                graph_builder = StateGraph(State)
                graph_builder.add_node("retrieve", retrieve)
                graph_builder.add_node("form_query", form_query)
                graph_builder.add_node("generate", generate)
                graph_builder.add_node("classify", classify)
                graph_builder.add_node("dont_know", dont_know)
                graph_builder.add_node("anonymize", anonymize)
                graph_builder.add_node(
                    "contact_customer_support", contact_customer_support
                )
                graph_builder.add_edge(START, "anonymize")
                graph_builder.add_edge("anonymize", "classify")
                # If pipeline should only run til classify node
                if os.environ.get("ONLY_CLASSIFY") == "false":
                    graph_builder.add_conditional_edges(
                        "classify", path=classify_path_function
                    )
                graph_builder.add_edge("retrieve", "generate")
                connection_pool = create_db_manager().conn_pool
                # Add checkpointer function from langgraph
                if connection_pool:
                    await connection_pool.open()
                    checkpointer = AsyncPostgresSaver(connection_pool)  # type: ignore
                    await checkpointer.setup()
                    self._graph = graph_builder.compile(checkpointer=checkpointer)
            except Exception as e:
                raise e
        return self._graph
