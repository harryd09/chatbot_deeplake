from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
from dotenv import load_dotenv
import os
import requests
from typing import List
from langchain.embeddings.base import Embeddings


# Custom embeddings class for OpenRouter
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


print("Loading environment variables...")
load_dotenv()
# get the environment variables
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")
ACTIVELOOP_ORG_ID = os.getenv("ACTIVELOOP_ORG_ID")
# Use separate key for embeddings if provided
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", OPENAI_API_KEY)
print("Environment variables loaded.")

# we'll use information from the following articles
urls = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-download-gif-twitter/",
    "https://beebom.com/how-use-chatgpt-linux-terminal/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-save-instagram-story-with-music/",
    "https://beebom.com/how-install-pip-windows/",
    "https://beebom.com/how-check-disk-usage-linux/",
]

# use the SeleniumURLLoader to load the articles
print(f"Loading {len(urls)} URLs with Selenium (this may take a few minutes)...")
import warnings

warnings.filterwarnings("ignore")
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()
print(f"Loaded {len(docs_not_splitted)} documents.")

# use the CharacterTextSplitter to split the documents into chunks
print("Splitting documents into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)
print(f"Created {len(docs)} document chunks.")

# use the OpenRouterEmbeddings to embed the chunks
print("Initializing embeddings via OpenRouter...")
embeddings = OpenRouterEmbeddings(
    api_key=OPENROUTER_API_KEY, model="openai/text-embedding-3-large"
)
print("Embeddings initialized.")

# create DeepLake data set
my_activeloop_org_id = ACTIVELOOP_ORG_ID
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# create DeepLake data set
print(f"Creating DeepLake dataset at {dataset_path}...")
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
print("DeepLake dataset created.")

# add the documents to the data set
print(f"Adding {len(docs)} documents to DeepLake (this may take a while)...")
db.add_documents(docs)
print("Documents added to DeepLake.")

# query the data set
print("\nQuerying: 'How to check disk usage in Linux?'")
results = db.similarity_search("How to check disk usage in Linux?")
print("\n=== Search Results ===")
print(results[0].page_content)
