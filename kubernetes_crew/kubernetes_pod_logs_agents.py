from crewai import Agent, Task, Crew, Process
from tools.CustomTools import CustomTools
from langchain_openai import ChatOpenAI
from phoenix.trace.langchain import LangChainInstrumentor
from kubernetes_pod_health_agents import pod_status_agent, fetch_pod_status_task
import phoenix as px
import os


session = px.launch_app()
LangChainInstrumentor().instrument()

os.environ["OPENAI_API_KEY"] = ""  # platform.openai key
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0.3)

# Creating a senior developer agent with kubernetes expertise
pod_name_agent = Agent(
    role='Kubernetes pod names agent',
    goal='fetch the pod names from the default namespace',
    verbose=True,
    memory=True,
    backstory=(
        "As a kubernetes developer agent with kubernetes expertise"
        "your job is to fetch the pod names and return the pod_name as array."
    ),
    tools=[CustomTools.fetch_pod_names],
    allow_delegation=True,
    llm=llm
)

pod_log_agent = Agent(
    role='Kubernetes pod log extractor',
    goal='fetch the pod log for a given pod name: {pod_name}',
    verbose=True,
    memory=True,
    backstory=(
        "As a kubernetes developer agent with kubernetes expertise"
        "your job is to fetch the pod logs and return the logs as single string "
        "only if the status of the pod is `Running`."
    ),
    tools=[CustomTools.fetch_pod_logs],
    allow_delegation=True,
    llm=llm
)

# pod name task
fetch_pod_names_task = Task(
    description=(
        "identify the pod for {pod} and return the pod name in the form of array"
    ),
    expected_output="A json array object with pod_name, example of the response as below \n"
                    "['pod_name': 'sample-asdasd-asda-asdasd'].",
    agent=pod_name_agent,
)

# pod log task
fetch_pod_logs_task = Task(
    description=(
        "extract the pod logs for {pod_name} and return the pod log in the form of string"
    ),
    expected_output="A json object with pod_name, log, example of the response as below \n"
                    "{"
                    "'pod_name': 'sample-asdasd-asda-asdasd', "
                    "'pod_status': 'RUNNING', "
                    "'log': 'actual log extracted by the agent'"
                    "}.",
    agent=pod_log_agent,
)

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[pod_name_agent, pod_status_agent, pod_log_agent],
    tasks=[fetch_pod_names_task, fetch_pod_status_task, fetch_pod_logs_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(px.active_session().url)
print(result)
