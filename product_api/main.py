from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from database.DB import DBUtil
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)

if os.getenv("DEBUG_LOG_LEVEL") == "true":
    logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
DBUtil.connect()


@app.get("/product/{product_name}")
def fetch_product_details(product_name: str):
    rows = DBUtil.query_product(product_name=product_name)
    return {'data': rows}


