from crewai_tools import tool
import requests
import re
import json


class CustomTools:

    @staticmethod
    @tool("fetch_pod_status")
    def fetch_pod_status() -> str:
        """ this is a tool to be used to check the pod status in kubernetes"""
        url = "http://localhost:8080/api/v1/namespaces/default/pods"
        response = requests.get(url=url)

        return response.json()

    @staticmethod
    @tool("fetch_pod_names")
    def fetch_pod_names() -> str:
        """ this is a tool to be used to fetch the pod names from kubernetes"""
        url = "http://localhost:8080/api/v1/namespaces/default/pods"
        response = requests.get(url=url)

        data = response.json()

        # Regex pattern you are looking for in the 'name' key of 'metadata'
        pattern = r'products-api'

        # Extract objects where 'name' in 'metadata' matches the regex pattern
        filtered_data = [
            item for item in data["items"] if re.search(pattern, item['metadata']['name'])
        ]

        return json.dumps(filtered_data, indent=2)

    @staticmethod
    @tool("fetch_pod_logs")
    def fetch_pod_logs(pod_name) -> str:
        """ this is a tool to be used to fetch the pod logs from kubernetes"""
        url = f"http://localhost:8080/api/v1/namespaces/default/pods/{pod_name}/log"
        response = requests.get(url=url)
        return response.text
