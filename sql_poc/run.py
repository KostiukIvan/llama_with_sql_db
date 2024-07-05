from llama_index.core.indices import GPTSQLStructStoreIndex
from sqlalchemy import create_engine
from llama_index.core.utilities.sql_wrapper import SQLDatabase
import pandas as pd
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.readers import StringIterableReader
from llama_index.readers.file import PandasCSVReader, UnstructuredReader

prompt = lambda x: f"""{x}
If questoin is about total amount, provide total amout by adding everything.
If the question is about days range, provide exect calculation how many days ago it happened and what date it was. Could you SQL for that.
Today's date is 2024-06-07 YYYY-MM-DD
Do not use strftime to date columns, use it as a strings. No formating.
Columns description:
1.Trans_date, SQL DATE format - date when transaction was done. When comparing the column's values to other values/strings, convert those values/strings to DATE format. Ex: DATE('YYYY-MM-DD'). Do not make any transformations with the 'Trans_date' column values, use it as it is.
3.Description, SQL string format - broad description of the transation, could be unstructured text. Use lowercase for data extraction. If you need to find an organization name look in this column!
4.Type, SQL string format -  it's generic type, that provides information about to what category transaction belongs. There is no organization names!
5.Amount, SQL float format - float number that represents amount of spent money in the transaction."""

file_path = 'data/estatements.csv'

def get_df_documents(df):
    # Function to create a single text representation of each row
    def create_text_representation(row):
        return ' | '.join([f"{col}: {row[col]}" for col in row.index])

    # Load CSV file into a DataFrame
    df = pd.read_csv(file_path, delimiter=',')

    # Create a 'text' column that summarizes all the information in each row
    df['text'] = df.apply(create_text_representation, axis=1)
    # Create the GPTSQLStructStoreIndex
    return StringIterableReader().load_data(
        texts=["\n".join([row['text'] for _, row in df.iterrows()])]
    )

def get_df_documentes_2():
    reader = PandasCSVReader()
    return reader.load_data(file_path)

def preproces_df(df):
    # Strip any leading/trailing spaces from Spend_Categories
    for col in df.columns:
        if col != "Amount":
            df[col] = df[col].str.strip()
            df = df.rename(columns={col: col.strip()})

    # Convert 'Trans_date' and 'Post_date' to actual date format
    df['Trans_date'] = pd.to_datetime(df['Trans_date'] + ', 2024')  # Assuming the year is 2022
    df['Post_date'] = pd.to_datetime(df['Post_date'] + ', 2024')    # Assuming the year is 2022

    return df


# Load CSV file into a DataFrame
df = pd.read_csv(file_path, delimiter=',')
df = preproces_df(df)
documents = get_df_documentes_2()

# Create an SQLite in-memory database
engine = create_engine('sqlite://', echo=False)

# Load DataFrame into the SQLite database
df.to_sql('ESTAT', con=engine, index=False, if_exists='replace')

# Initialize the SQLDatabase object
sql_database = SQLDatabase(engine)
# print(sql_database.get_single_table_info(table_name="ESTAT"))
# print(sql_database.run_sql("SELECT SUM(Amount) AS Total_Spent \nFROM ESTAT \nWHERE Trans_date >= DATE('2022-03-30') \nAND LOWER(Description) LIKE '%starbucks%';"))
# print(sql_database.run_sql("SELECT SUM(Amount) AS Total_Spent FROM ESTAT WHERE Trans_date >= DATE('2024-03-30') AND LOWER(Description) LIKE '%starbucks%';"))
print(sql_database.run_sql("SELECT SUM(Amount) AS TotalAmount FROM ESTAT WHERE Trans_date >= '2024-03' AND Description LIKE '%starbucks%' OR Type LIKE '%starbucks%';"))

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0, temperature=0.0, json_mode=True)

index = GPTSQLStructStoreIndex(documents=documents,sql_database=sql_database, table_name='ESTAT')

# Convert the index to a query engine
query_engine = index.as_query_engine()

# Example natural language query
# nl_query = "What are the entries where Amount is greater than 100?"
# nl_query = prompt("How much did I spend on STARBUCKS in the last 100 days?")
nl_query = prompt("How much did I pay Starbucks in last 3 months?")
#nl_query = "Total I spent on Starbucks since March 30?"
# SELECT SUM(Amount) AS Total_Spent FROM ESTAT WHERE Trans_date >= DATE('2022-03-30') AND LOWER(Description) LIKE '%starbucks%'

#nl_query = "Total I spent on Starbucks since March 30? NOTE! Before creating SQL query, take a look on a table and find where asked data are located and use appropriate columns for that. Even the SQL query would be more complex, correctness is our main priority!!!"
# nl_query = prompt("""Total I spent on Starbucks since March 30?""")


# Execute the natural language query
response = query_engine.query(nl_query)
print("Response:", response)

