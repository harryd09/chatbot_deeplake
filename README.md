# Customer Support Chatbot Example

A production-ready chatbot that answers customer support questions using RAG (Retrieval-Augmented Generation) with LangChain, DeepLake, and OpenRouter.

## Overview

This project demonstrates how to build a customer support chatbot that:
- Scrapes content from multiple web pages using Selenium
- Stores document embeddings in DeepLake vector database
- Retrieves relevant context for user queries
- Generates accurate answers using LLM with retrieved context

## Project Structure

- `load_data.py` - Scrapes URLs, creates embeddings, and stores them in DeepLake
- `chat.py` - Loads the vector database and answers user queries
- `pyproject.toml` - Project dependencies managed with uv

## Setup

### Prerequisites

- Python 3.11+
- uv package manager
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))
- ActiveLoop account ([Sign up here](https://www.activeloop.ai/))

### Installation

1. Clone the repository and navigate to the project directory

2. Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken unstructured selenium
```

3. Create a `.env` file in the project root:
```bash
# OpenRouter API Key (for LLM and embeddings)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key

# ActiveLoop credentials
ACTIVELOOP_TOKEN=your-activeloop-token
ACTIVELOOP_ORG_ID=your-activeloop-org-id

# Optional: Custom API base URL for chat (defaults to OpenRouter)
OPENAI_API_BASE_URL=https://openrouter.ai/api/v1
```

## Usage

### 1. Build the Knowledge Base

Run `load_data.py` to scrape URLs and create the vector database:

```bash
uv run load_data.py
```

This will:
- Load 8 web pages using Selenium
- Split documents into chunks
- Generate embeddings using `openai/text-embedding-3-large`
- Store everything in DeepLake at `hub://your-org-id/langchain_course_customer_support`

**Note:** This may take several minutes depending on the number of URLs and document size.

### 2. Query the Chatbot

Run `chat.py` to ask questions:

```bash
uv run chat.py
```

The chatbot will:
- Load the DeepLake vector database
- Search for relevant context
- Generate an answer using the LLM (currently using `deepseek/deepseek-v3.2`)

To ask different questions, edit the `query` variable in `chat.py`:

```python
query = "Your question here?"
```

## Configuration

### Changing the LLM Model

Edit `chat.py` and modify the model parameter:

```python
llm = ChatOpenAI(
    model="openai/gpt-4",  # or any OpenRouter model
    temperature=0,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)
```

Available models on OpenRouter:
- `openai/gpt-4` - Best quality, higher cost
- `openai/gpt-3.5-turbo` - Good quality, lower cost
- `deepseek/deepseek-v3.2` - Fast and cost-effective
- [See all models](https://openrouter.ai/models)

### Changing URLs

Edit the `urls` list in `load_data.py`:

```python
urls = [
    'https://example.com/article1',
    'https://example.com/article2',
    # Add more URLs...
]
```

### Changing Embedding Model

Edit the `OpenRouterEmbeddings` initialization in both files:

```python
embeddings = OpenRouterEmbeddings(
    api_key=OPENROUTER_API_KEY,
    model="openai/text-embedding-3-small"  # Cheaper alternative
)
```

## Architecture

See [AGENTS.md](./AGENTS.md) for detailed information about the system architecture and how the components work together.

## Troubleshooting

### Selenium Issues

If Selenium fails to load pages, you may need to install browser drivers:
- Chrome: `brew install chromedriver` (macOS)
- Firefox: `brew install geckodriver` (macOS)

### DeepLake Connection Issues

Ensure your `ACTIVELOOP_TOKEN` is valid and you have access to the organization specified in `ACTIVELOOP_ORG_ID`.

### API Rate Limits

OpenRouter has rate limits. If you hit them, consider:
- Adding delays between requests
- Using a different model with higher limits
- Upgrading your OpenRouter plan

## License

This project is based on examples from "Building LLMs for Production" book.

## Contributing

Feel free to open issues or submit pull requests with improvements!
