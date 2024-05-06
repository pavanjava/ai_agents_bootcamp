from crewai import Agent
from langchain_openai import ChatOpenAI
from core.Tools import QdrantTools
from dotenv import load_dotenv, find_dotenv
import os


class QdrantAgents:
    def __init__(self):
        _ = load_dotenv(find_dotenv())
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # platform.openai key
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_LLM"), temperature=0.3)

    def vectorizer_agent(self, prompt):
        return Agent(
            role='vectorize any given query pod names agent',
            goal=f'fetch the vector embedding for the given {prompt}',
            verbose=True,
            memory=True,
            backstory=(
                f"As a senior vector database engineer agent with qdrant vector database expertise "
                f"your job is to convert the given {prompt} into vector embedding."
            ),
            tools=[QdrantTools.vectorize_query],
            allow_delegation=True,
            llm=self.llm
        )

    def vector_search_agent(self):
        return Agent(
            role="vector search agent",
            goal="search the qdrant vector store for a given vector embedding",
            verbose=True,
            memory=True,
            backstory=("you need to search the qdrant vector database to fetch the top 2 similar data"
                       "by using the embedding from the previous agent"),
            tools=[QdrantTools.search_qdrant],
            allow_deligation=False,
            llm=self.llm
        )
