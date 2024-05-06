import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = ""  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "" # platform.openai key

search_tool = SerperDevTool()

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
    role='Writer',
    goal='Create a deployment.yaml to deploy a container on kubernetes',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topic of kubernetes deployment, you craft"
        "correct deployment.yaml that has placeholder for container_name, ports, environment_variables etc"
        "code yaml file that can directly run with kubectl command."
    ),
    tools=[search_tool],
    allow_delegation=False
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Compose an insightful deployment.yaml which is a standard deployment.yaml file targeted for kubernetes"
        "Focus on the yaml file with the container_name, environment_variables, resources with limits,"
        " service section, ports (port, targetPort, nodePort). This deployment.yaml should be run "
        "using kubectl with no errors."
    ),
    expected_output='A standard deployment.yaml along with container_name, environment_variables, '
                    'resources with limits, service section with out any junk characters in it',
    agent=writer,
    async_execution=False,
    output_file='deployment.yaml'  # Example of output customization
)

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[writer],
    tasks=[write_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(result)

