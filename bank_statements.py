import pandas as pd
import hashlib
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.readers import StringIterableReader
from llama_index.readers.file import PandasCSVReader, PagedCSVReader

# Function to create a single text representation of each row
def create_text_representation(row):
    return ' | '.join([f"{col}: {row[col]}" for col in row.index])

# Load CSV data
df = pd.read_csv("estatements.csv", sep=',')

# Create a 'text' column that summarizes all the information in each row
df['text'] = df.apply(create_text_representation, axis=1)

# Setup the embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Setup the large language model
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Create a list of document strings from the 'text' column
# documents = [SimpleDocument(id_=index, text=row['text']) for index, row in df.iterrows()]
documents = StringIterableReader().load_data(
    texts=["\n".join([row['text'] for _, row in df.iterrows()])]
)
# Create index from documents
def get_df_documentes_2():
    reader = PandasCSVReader()
    return reader.load_data('estatements.csv')
documents = get_df_documentes_2()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Example query about the bank statement
response = query_engine.query("What were total spendings between Trans_date=Mar 21 and Trans_date=Mar 25?")
print(response)
response = query_engine.query("What transaction was the most expensive and its amount?")
print(response)
response = query_engine.query("What transaction was the least expensive and its amount?")
print(response)
