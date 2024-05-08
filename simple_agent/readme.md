## LangGraph Search Agent

#### The code in main.py is a Python script that creates a simple question-answering system using the LangChain library and the OpenAI language model. Here's an explanation of what the code does:
- Purpose: The purpose of this code is to create a system that can answer a given question by first searching for relevant information on the internet using the Google Search API, and then generating a concise answer using the OpenAI language model.
- Input: The input to this system is a question in the form of a string, provided as part of the input dictionary.
- Output: The output of the system is a concise answer to the given question, along with some metadata about the response.
- Logic and Algorithm:

  - The code defines two main functions: retrieve_results and generate_answer.
  - The retrieve_results function takes the input question and uses the Google Search API to search for relevant information on the internet. It returns the search results as part of the messages list.
  - The generate_answer function takes the search results and the original question, and passes them to the OpenAI language model. The model is instructed to generate a concise answer (less than 30 words) based on the given context.
  - The code sets up a workflow using the StateGraph class from the langgraph library. This workflow alternates between the retrieve_results and generate_answer functions, allowing the system to iteratively search for information and generate answers.
  - The workflow is compiled into an executable application using the app = workflow.compile() line.
  - The app.stream function is used to execute the workflow and print the intermediate outputs from each step.
  
- Finally, the app.invoke function is called to get the final answer and its metadata.