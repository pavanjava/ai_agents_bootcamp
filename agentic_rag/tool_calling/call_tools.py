from utils.helper import get_docs, get_nodes, get_qdrant_vector_index
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilters


def add(x: int, y: int) -> int:
    """Adds two integers and returns the response"""
    return x + y


def a_plus_b_squared(a: int, b: int) -> int:
    """this function operates on two integers"""
    return (a + b) * (a + b)


add_tool = FunctionTool.from_defaults(fn=add)
a_plus_b_squared_tool = FunctionTool.from_defaults(fn=a_plus_b_squared)

llm = Settings.llm
response = llm.predict_and_call(tools=[add_tool, a_plus_b_squared_tool],
                                user_msg="Tell me the output of a plus b squared where a=2, b=3?",
                                verbose=True)

print(response)

################## Vector Query Agent #################
docs = get_docs()
nodes = get_nodes(docs=docs)
vector_index = get_qdrant_vector_index(nodes=nodes)
query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some top results of AutoGen? give the response as bullet points",
)

print(response)
print(response.metadata)
print(response.source_nodes)
