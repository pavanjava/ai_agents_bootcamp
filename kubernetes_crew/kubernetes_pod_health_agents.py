from crewai import Agent, Task, Crew, Process
from tools.CustomTools import CustomTools
import os

os.environ["OPENAI_API_KEY"] = ""  # platform.openai key

# Creating a senior developer agent with kubernetes expertise
pod_status_agent = Agent(
    role='Kubernetes pod health monitoring',
    goal='check the pod status in kubernetes',
    verbose=True,
    memory=True,
    backstory=(
        "As a senior developer agent with kubernetes expertise"
        "your job is to check the status of the pods and return the pod_name and the respective status."
    ),
    tools=[CustomTools.fetch_pod_status],
    allow_delegation=True
)

# Research task
fetch_pod_status_task = Task(
    description=(
        "identify the status of the pod and return the pod name along with the status of the pod"
    ),
    expected_output="A json object with pod_name and status example of the response as below \n"
                    "{'pod_name': 'sample', 'status': 'RUNNING'}.",
    agent=pod_status_agent,
)

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[pod_status_agent],
    tasks=[fetch_pod_status_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()

print(result)
