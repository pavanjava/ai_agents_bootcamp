from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from agentic_vector_search import crew_start
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)

if os.getenv("DEBUG_LOG_LEVEL") == "true":
    logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


@app.get("/crew-kickoff")
def crew_kickoff():
    crew_start()


