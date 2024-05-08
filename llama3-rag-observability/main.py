from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer)
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core import set_global_handler
import qdrant_client
import phoenix as px
import logging

px.launch_app()
set_global_handler("arize_phoenix")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load the local data directory and chunk the data for further processing
docs = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"]).load_data(show_progress=True)
text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

# Create a local Qdrant vector store
logger.info("initializing the vector store related objects")
client = qdrant_client.QdrantClient(url="http://localhost:6333/")
vector_store = QdrantVectorStore(client=client, collection_name="research_papers")

# local vector embeddings model
logger.info("initializing the OllamaEmbedding")
embed_model = OllamaEmbedding(model_name='mxbai-embed-large', base_url='http://localhost:11434')

logger.info("initializing the global settings")
Settings.embed_model = embed_model
Settings.llm = Ollama(model="llama3", base_url='http://localhost:11434')
Settings.transformations = [text_parser]

text_chunks = []
doc_ids = []
nodes = []

logger.info("enumerating docs")
for doc_idx, doc in enumerate(docs):
    curr_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(curr_text_chunks)
    doc_ids.extend([doc_idx] * len(curr_text_chunks))

logger.info("enumerating text_chunks")
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(text=text_chunk)
    src_doc = docs[doc_ids[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

logger.info("enumerating nodes")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode=MetadataMode.ALL)
    )
    node.embedding = node_embedding

logger.info("initializing the storage context")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info("indexing the nodes in VectorStoreIndex")
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    transformations=Settings.transformations,
)

logger.info("initializing the VectorIndexRetriever with top_k as 2")
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
response_synthesizer = get_response_synthesizer()
logger.info("creating the RetrieverQueryEngine instance")
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
logger.info("creating the HyDEQueryTransform instance")
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)

logger.info("retrieving the response to the query")

# Start a loop to continually get input from the user
while True:
    # Get a query from the user
    user_query = input("Enter your query (type 'bye' to exit): ")

    # Check if the user wants to terminate the loop
    if user_query.lower() == "bye":
        break

    response = hyde_query_engine.query(str_or_query_bundle=user_query)
    print(px.active_session().url)
    print(response)
    client.close()
