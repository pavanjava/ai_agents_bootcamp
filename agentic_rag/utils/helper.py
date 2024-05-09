import qdrant_client
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import os

from llama_index.vector_stores.qdrant import QdrantVectorStore

Settings.llm = Ollama(model="llama3:latest", base_url="http://localhost:11434", request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest", base_url="http://localhost:11434", )
Settings.chunk_size = 1024
Settings.chunk_overlap = 128


def set_openai_api_key_in_environment():
    os.environ["OPENAI_API_KEY"] = "sk-proj-HwSw1Aa8HSZ1ZnXON7xJT3BlbkFJJPVp0kE2OBISFg3RJPDE"


def get_docs() -> list[Document]:
    docs: list[Document] = SimpleDirectoryReader(input_dir="../data", required_exts=['.pdf']).load_data()
    return docs


def get_nodes(docs: list[Document]) -> list[BaseNode]:
    # split and create nodes
    splitter = SentenceSplitter(chunk_size=Settings.chunk_size)
    nodes: list[BaseNode] = splitter.get_nodes_from_documents(documents=docs)
    return nodes


def get_qdrant_vector_index(nodes: list[BaseNode]) -> VectorStoreIndex:
    # Define Summary Index and Vector Index over the nodes
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name="autogen_research_paper")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # indexes
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,
                                    transformations=Settings.transformations)
    return vector_index
