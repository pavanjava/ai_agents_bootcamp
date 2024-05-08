from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import os

# setup environment variables - prerequisites
os.environ["SERPER_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that retrieves information from serper
def retrieve_results(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    search_serper_api_wrapper = GoogleSerperAPIWrapper()
    search_tool = Tool(name="Search", func=search_serper_api_wrapper.run, description="Get an answer");
    search_response = search_tool.invoke(last_message)
    return {"messages": [search_response]}


# Define the function that calls the LLM to generate
def generate_answer(state):
    messages = state['messages']
    last_message = messages[-1]
    prompt_input_messages = [
        SystemMessage(
            content="You are a helpful assistant that can understand the given message, "
                    "context and generate an answer in less than 30 words only."
        ),
        HumanMessage(
            content=last_message
        ),
    ]
    generated_response = model.invoke(prompt_input_messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [generated_response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("retriever", retrieve_results)
workflow.add_node("generate", generate_answer)

workflow.add_edge("retriever", "generate")

# Set the entrypoint as `agent` where we start
workflow.set_entry_point("retriever")
workflow.set_finish_point("generate")

app = workflow.compile()

input = {"messages": ["What is the most expensive car Elon musk drives?"]}

for output in app.stream(input):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

response = app.invoke(input)
print(response['messages'][1])
print(response['messages'][2].response_metadata)
