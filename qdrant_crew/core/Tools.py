from typing import List
from crewai_tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams
from dotenv import load_dotenv, find_dotenv
import requests
import os


class QdrantTools:
    _ = load_dotenv(find_dotenv())
    # Create a Qdrant client instance
    client = QdrantClient(url=os.getenv("QDRANT_LOCAL"))

    @staticmethod
    @tool("vectorize_query")
    def vectorize_query(query: str) -> str:
        """ this is a tool to be used to vectorize the given query"""
        url = os.getenv("LOCAL_EMBEDDINGS")

        payload = {
            "model": os.getenv("LOCAL_EMBEDDINGS_MODEL"),
            "prompt": query
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)

        return response.json()

    @staticmethod
    @tool("search_qdrant")
    def search_qdrant(vector: List) -> str:
        """ this is a tool to be used to fetch the similar vectors from qdrant database"""
        # Create a search request
        result: list[ScoredPoint] = QdrantTools.client.search(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            query_vector=vector,
            search_params=SearchParams(hnsw_ef=128, exact=False),
            limit=2  # number of nearest vectors
        )
        return result
