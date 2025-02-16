from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


documents = SimpleDirectoryReader("data").load_data()

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()
response = query_engine.query("Generate 10 possible SQL queries to run on the data")
print(response)
# response = query_engine.query("What transaction was the most expensive and amount?")
# print(response)
# response = query_engine.query("What transaction was the least expensive and amount?")
# print(response)