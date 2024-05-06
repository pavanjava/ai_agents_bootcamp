from core.Agents import QdrantAgents
from core.Tasks import QdrantTasks
from crewai import Crew, Process
from phoenix.trace.langchain import LangChainInstrumentor
import phoenix as px
import warnings

warnings.filterwarnings('ignore')

session = px.launch_app()
LangChainInstrumentor().instrument()

qdrant_agents = QdrantAgents()
qdrant_tasks = QdrantTasks()

prompt = ('That is where the accuracy matters the most. And in this case, '
          'Qdrant has proved just commendable in giving excellent search results.')

vectorizer_agent = qdrant_agents.vectorizer_agent(prompt=prompt)
vector_search_agent = qdrant_agents.vector_search_agent()

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[vectorizer_agent, vector_search_agent],
    tasks=[qdrant_tasks.vectorize_prompt_task(agent=vectorizer_agent, prompt=prompt),
           qdrant_tasks.vector_search_task(agent=vector_search_agent)],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)


# Starting the task execution process with enhanced feedback
def crew_start():
    result = crew.kickoff()
    print(px.active_session().url)
    print(result)
