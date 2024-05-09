import qdrant_client
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from utils.helper import set_openai_api_key_in_environment
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex, StorageContext
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# set_openai_api_key_in_environment()


class AgenticRouter:
    def __init__(self):
        Settings.llm = Ollama(model="llama3:latest", base_url="http://localhost:11434", request_timeout=300)
        Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest", base_url="http://localhost:11434", )
        Settings.chunk_size = 1024

        # loading documents
        docs = SimpleDirectoryReader(input_dir="../data", required_exts=[".pdf"]).load_data()

        # split and create nodes
        splitter = SentenceSplitter(chunk_size=Settings.chunk_size)
        self.nodes = splitter.get_nodes_from_documents(documents=docs)

    def execute(self, query: str):
        # Define Summary Index and Vector Index over the nodes
        client = qdrant_client.QdrantClient(host="localhost", port=6333)
        vector_store = QdrantVectorStore(client=client, collection_name="autogen_research_paper")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # indexes
        vector_index = VectorStoreIndex(nodes=self.nodes, storage_context=storage_context,transformations=Settings.transformations)
        summary_index = SummaryIndex(nodes=self.nodes)

        # define query engines on the index and set metadata
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",use_async=True)
        vector_query_engine = vector_index.as_query_engine()

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions related to AutoGen"
            ),
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context from the AutoGen paper."
            ),
        )

        # define the router engine on top of tools
        router_query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
            ],
            verbose=True
        )

        # chat with agent
        response = router_query_engine.query(query)
        print(str(response))
        print(response.metadata)

        # response = router_query_engine.query(
        #     "what is Retrieval-Augmented Code Generation and Question Answering"
        # )
        # print(str(response))


agentic_router = AgenticRouter()
agentic_router.execute("what is Retrieval-Augmented Code Generation and Question Answering")



