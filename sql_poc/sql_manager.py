from llama_index.core.indices import GPTSQLStructStoreIndex
from sqlalchemy import create_engine
from llama_index.core.utilities.sql_wrapper import SQLDatabase
import pandas as pd
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.readers import StringIterableReader
from llama_index.readers.file import PandasCSVReader, UnstructuredReader
import json

promt = lambda x: f"""
Example Question: What is the total spending on LCBO in April? 
Example Response: SELECT SUM(Amount) AS Total_Spending_LCBO_April
                FROM transactions
                WHERE Description LIKE '%LCBO%' AND Trans_date BETWEEN '2024-04-01' AND '2024-04-30';


Question: {x}
Response: ???"""

class Manager:
    def __init__(self, file_path: str, temperature = 0.0) -> None:
        self._file_path = file_path
        self._documents = None
        self._df = None
        self._engine = None
        self._sql_db = None
        self._index = None
        self._table_name = 'transactions'
        self._query_engine = None

        # Load CSV file into a DataFrame
        self._df = pd.read_csv(self._file_path, delimiter=',')
        self._preproces_df()
        self._documents = self._get_pandas_CSV_reader_document()

        # Create an SQLite in-memory database
        self._engine = create_engine('sqlite://', echo=False)

        # Load DataFrame into the SQLite database
        self._df.to_sql(self._table_name, con=self._engine, index=False, if_exists='replace')

        # Initialize the SQLDatabase object
        self._sql_db = SQLDatabase(self._engine)

        # nomic embedding model
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # ollama
        Settings.llm = Ollama(model="llama3", request_timeout=360.0, temperature=temperature, json_mode=True)

        self._index = GPTSQLStructStoreIndex(documents=self._documents,sql_database=self._sql_db, table_name=self._table_name)

        # Convert the index to a query engine
        self._query_engine = self._index.as_query_engine(sql_only=True)

    def _get_pandas_CSV_reader_document(self):
        reader = PandasCSVReader()
        return reader.load_data(self._file_path)

    def _preproces_df(self):
        # Strip any leading/trailing spaces from Spend_Categories
        for col in self._df.columns:
            if col != "Amount":
                self._df[col] = self._df[col].str.strip()
                self._df = self._df.rename(columns={col: col.strip()})

        # Convert 'Trans_date' and 'Post_date' to actual date format
        self._df['Trans_date'] = pd.to_datetime(self._df['Trans_date'] + ', 2024')  # Assuming the year is 2022

    def query_llm(self, query):
        
        response = self._query_engine.query(promt(query))
        try:
            sql = json.loads(response.response)["SQLQuery"]
        except Exception as e:
            print(e)
            return None
        
        return sql

    def query_db(self, sql):
        try:
            return self._sql_db.run_sql(sql)
        except Exception as e:
            print(e)
            return None


if __name__=="__main__":
    m = Manager(file_path='data/fake_estatements.csv')
    sql = m.query_llm("Calculate the average amount spent on McDonald’s in April?")
    print(sql)
    db = m.query_db(sql)
    print(db)