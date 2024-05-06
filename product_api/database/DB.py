from utils.GenericUtils import convert_to_tuple
from dotenv import load_dotenv, find_dotenv
import psycopg2
import os

_ = load_dotenv(find_dotenv())


class DBUtil:
    conn = None

    @staticmethod
    def connect() -> None:
        # Establish a connection to the database
        DBUtil.conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=5432,
            database="postgres",
            user="root",
            password="root"
        )

    @staticmethod
    def query_product(product_name):
        # Create a cursor object
        cur = DBUtil.conn.cursor()

        # SQL query to fetch data
        product_name = product_name  # Example placeholder for product ID
        input = convert_to_tuple(product_name)
        query = f"SELECT * FROM products WHERE product_name in {input};"
        print(query)
        # Execute the query
        cur.execute(query, (product_name,))

        # Fetch the results
        rows = cur.fetchall()

        # Get column names from the cursor description
        column_names = [desc[0] for desc in cur.description]

        # Printing column names and values for each row
        product_details = []
        for row in rows:
            product_details.append(zip(column_names, row))

        return product_details
