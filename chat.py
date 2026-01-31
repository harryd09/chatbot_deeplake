from langchain import PromptTemplate
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DeepLake
from langchain.schema import HumanMessage
import requests
from typing import List
from langchain.embeddings.base import Embeddings


# Custom embeddings class for OpenRouter (same as in main.py)
class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-large"):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://openrouter.ai/api/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using OpenRouter API."""
        response = requests.post(
            self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
            },
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using OpenRouter API."""
        response = requests.post(
            self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
            },
            json={"model": self.model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")
ACTIVELOOP_ORG_ID = os.getenv("ACTIVELOOP_ORG_ID")

# lets write a prompt for a customer support chatbot that answers questions using information
# extracted from our knowledge base

template = """
You are an exceptional customer support chatbot that gently answers questions.

You know the following context information:

{chunks_formatted}

Answer to the following question from a customer. Use only information from
previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt_template = PromptTemplate(
    template=template, input_variables=["chunks_formatted", "query"]
)

# Load the DeepLake database
print("Loading DeepLake database...")
embeddings = OpenRouterEmbeddings(
    api_key=OPENROUTER_API_KEY, model="openai/text-embedding-3-large"
)
dataset_path = f"hub://{ACTIVELOOP_ORG_ID}/langchain_course_customer_support"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=True)
print("Database loaded.")

# the full pipeline# user questio
query = "How to check disk usage in linux?"
print(f"\nQuery: {query}")
# retrieve relevant chunks
print("Searching for relevant information...")
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]
# format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt_template.format(
    chunks_formatted=chunks_formatted, query=query
)
# generate answer
print("Generating answer...")
llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2",
    temperature=0,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)
# Use chat format instead of completion
messages = [HumanMessage(content=prompt_formatted)]
answer = llm(messages)
print("\n=== Answer ===")
print(answer.content)
