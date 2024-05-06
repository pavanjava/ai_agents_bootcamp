from core.Agents import QdrantAgents
from crewai import Task


class QdrantTasks:
    def __init__(self):
        self.qdrant_agents = QdrantAgents()

    def vectorize_prompt_task(self, agent, prompt):
        return Task(
            description=(
                f"identify the {prompt} and return the vector embedding of the respective {prompt}"
            ),
            expected_output="A json array object with vector embeddings, example of the response as below \n"
                            "[-0.09073494374752045,-0.22729796171188354,-0.2276609241962433,0.2960631847381592].",
            agent=agent
        )

    def vector_search_task(self, agent):
        return Task(
            description=(
                "identify the top 2 similar data from the database by using the embedding from previous task."
            ),
            expected_output="A json object with similar vectors and scores associated with them",
            agent=agent
        )
