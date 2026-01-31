# System Architecture & Agent Design

This document explains how the customer support chatbot works, including the RAG (Retrieval-Augmented Generation) architecture and the role of each component.

## Overview

The chatbot uses a **RAG architecture** to provide accurate answers based on a knowledge base rather than relying solely on the LLM's training data.

### Why RAG?

1. **Accuracy**: Answers are grounded in specific, verified documents
2. **Up-to-date**: Knowledge base can be updated without retraining the model
3. **Transparency**: We can trace answers back to source documents
4. **Cost-effective**: No need to fine-tune large language models

## System Components

### 1. Document Ingestion Pipeline (`load_data.py`)

```
Web Pages → Selenium → Document Loader → Text Splitter → Embeddings → Vector Database
```

#### Components:

**SeleniumURLLoader**
- Loads web pages dynamically (handles JavaScript-rendered content)
- Extracts text content from HTML
- Returns raw documents

**CharacterTextSplitter**
- Splits documents into manageable chunks (1000 characters)
- No overlap between chunks (`chunk_overlap=0`)
- Ensures chunks fit within embedding model context limits

**OpenRouterEmbeddings (Custom)**
- Converts text chunks into vector embeddings
- Uses `openai/text-embedding-3-large` (3072 dimensions)
- Communicates with OpenRouter's embeddings API
- Implements both batch (`embed_documents`) and single (`embed_query`) embedding

**DeepLake Vector Database**
- Stores document chunks and their embeddings
- Enables similarity search
- Cloud-hosted for easy sharing and collaboration
- Read-only mode in chat.py for concurrent access

### 2. Query & Response Pipeline (`chat.py`)

```
User Query → Embeddings → Similarity Search → Context Retrieval → Prompt + LLM → Answer
```

#### Components:

**Query Embedding**
- User's question is converted to the same embedding space
- Uses the same embedding model as document ingestion

**Similarity Search**
- Compares query embedding with stored document embeddings
- Returns top-K most relevant chunks (default: 4)
- Uses cosine similarity or L2 distance

**Prompt Template**
- Structures the retrieved context and user query
- Instructs LLM to answer only from provided context
- Prevents hallucination by constraining the LLM

**ChatOpenAI (LLM)**
- Generates natural language answer
- Uses chat completion format (not text completion)
- Currently uses `deepseek/deepseek-v3.2` via OpenRouter
- Temperature set to 0 for deterministic responses

## Data Flow

### Building the Knowledge Base

1. **URL Loading**
   ```python
   loader = SeleniumURLLoader(urls=urls)
   docs_not_splitted = loader.load()
   # Result: 8 documents (one per URL)
   ```

2. **Text Splitting**
   ```python
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   docs = text_splitter.split_documents(docs_not_splitted)
   # Result: 107 chunks
   ```

3. **Embedding Generation**
   ```python
   embeddings = OpenRouterEmbeddings(api_key=OPENROUTER_API_KEY)
   # Each chunk → 3072-dimensional vector
   ```

4. **Vector Storage**
   ```python
   db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
   db.add_documents(docs)
   # Stores 107 chunks with embeddings in DeepLake
   ```

### Answering Questions

1. **User Query**
   ```python
   query = "How to check disk usage in linux?"
   ```

2. **Semantic Search**
   ```python
   docs = db.similarity_search(query)
   # Returns 4 most relevant chunks
   ```

3. **Context Assembly**
   ```python
   chunks_formatted = "\n\n".join([doc.page_content for doc in docs])
   # Combines retrieved chunks into context
   ```

4. **Prompt Construction**
   ```python
   prompt_formatted = prompt_template.format(
       chunks_formatted=chunks_formatted,
       query=query
   )
   ```

   Example final prompt:
   ```
   You are an exceptional customer support chatbot...

   You know the following context information:
   [Retrieved chunk 1]

   [Retrieved chunk 2]

   [Retrieved chunk 3]

   [Retrieved chunk 4]

   Question: How to check disk usage in linux?

   Answer:
   ```

5. **LLM Generation**
   ```python
   messages = [HumanMessage(content=prompt_formatted)]
   answer = llm(messages)
   # LLM generates answer based on context
   ```

## Custom Components

### OpenRouterEmbeddings Class

We built a custom embeddings class because:

1. **Compatibility**: Old langchain (0.0.208) doesn't support OpenRouter natively
2. **API Format**: OpenRouter requires specific header format
3. **Model Naming**: OpenRouter uses `openai/model-name` format

```python
class OpenRouterEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding for multiple documents
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:3000",
            },
            json={"model": self.model, "input": texts}
        )
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        # Single embedding for queries
        # Similar implementation as above
```

## Key Design Decisions

### Why DeepLake?

- **Cloud-native**: No local database management
- **Version control**: Built-in versioning for datasets
- **Collaboration**: Easy sharing across team members
- **Scalability**: Handles millions of embeddings efficiently

### Why OpenRouter?

- **Model flexibility**: Access to multiple LLM providers
- **Cost optimization**: Choose models based on price/performance
- **Reliability**: Built-in fallbacks and load balancing
- **Single API**: Unified interface for different models

### Prompt Engineering

The prompt template includes important constraints:

1. **Role definition**: "exceptional customer support chatbot"
2. **Tone instruction**: "gently answers questions"
3. **Context grounding**: "Use only information from previous context"
4. **Anti-hallucination**: "Do not invent stuff"

These constraints ensure:
- Consistent tone across responses
- Factual accuracy (no made-up information)
- Professional customer service quality

## Performance Considerations

### Embedding Generation

- **Batch processing**: Documents are embedded in batches (default: 107 at once)
- **API calls**: Minimize calls by batching
- **Cost**: ~107 chunks × $0.00013/1K tokens (3-large) = ~$0.014 per rebuild

### Query Time

1. Embed query: ~100ms
2. Similarity search: ~50ms (DeepLake)
3. LLM generation: ~2-5s (model dependent)
4. **Total**: ~2-5 seconds per query

### Optimization Opportunities

1. **Cache frequent queries**: Store common Q&A pairs
2. **Smaller embeddings**: Use `text-embedding-3-small` (1536d vs 3072d)
3. **Faster LLM**: Switch to `gpt-3.5-turbo` for speed
4. **Streaming**: Implement streaming responses for better UX

## Extending the System

### Adding New Data Sources

1. Implement custom document loaders:
   - PDFLoader for PDF documents
   - CSVLoader for structured data
   - APILoader for REST APIs

2. Update `load_data.py` to include new sources
3. Rebuild the vector database

### Multi-turn Conversations

Add conversation memory:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# Include conversation history in prompts
```

### Answer Citations

Track source documents:

```python
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
```

### Evaluation & Monitoring

1. **Track metrics**:
   - Answer relevance
   - Response time
   - User satisfaction

2. **A/B testing**:
   - Different prompt templates
   - Different LLM models
   - Different chunk sizes

## Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **Rate limiting**: Implement request throttling
3. **Input validation**: Sanitize user queries
4. **Access control**: Use DeepLake's permission system

## Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [DeepLake Documentation](https://docs.activeloop.ai/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
