# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine built for a movie dataset (Hoopla). Originally created as an exploratory project to learn and experiment with RAG architectures, it provides a comprehensive suite of CLI tools to explore various search paradigms, including keyword-based search, semantic search, hybrid search (using Reciprocal Rank Fusion or weighted scores), and multimodal image-to-text search. It also features a RAG module to generate human-like, natural language answers based on the retrieved documents.

## Features

- **Keyword Search**: Fast and efficient exact-match searching utilizing BM25 and TF-IDF ranking algorithms.
- **Semantic Search**: Meaning-based search powered by `sentence-transformers`, complete with advanced text chunking (fixed-size and semantic boundaries) capabilities.
- **Hybrid Search**: Fuses BM25 and semantic scores using Reciprocal Rank Fusion (RRF) and weighted combinations to deliver highly relevant results. Includes query enhancement and re-ranking options.
- **Multimodal Search**: Search for movies using images as queries through CLIP image embeddings.
- **Image Description/Query Rewriting**: Enhances text queries by synthesizing visual information from images using Google's Gemini 2.5 Flash model.
- **Retrieval-Augmented Generation (RAG)**: Leverages `gemma-4-31b-it` (via Google GenAI API) to generate detailed, conversational, or summarized answers complete with source citations.
- **Evaluation Engine**: Benchmark search performance calculating Precision, Recall, and F1 scores against a golden dataset.

## Setup & Installation

### Prerequisites
- Python `>= 3.13`
- API Key for Google GenAI (Gemini)

### Installation

1. Navigate to the project directory:
   ```bash
   cd rag-search-engine
   ```

2. Install dependencies using `uv` (recommended) or `pip`:
   ```bash
   uv sync
   # OR
   pip install -e .
   ```

3. Configure Environment Variables:
   Copy the example environment file and add your Gemini API key.
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set:
   ```
   GEMINI_API_KEY="your_api_key_here"
   ```

## CLI Usage

The project offers multiple CLI entry points inside the `cli/` directory. Here are some common examples:

### Keyword Search
Build the inverted index and perform a BM25 search.
```bash
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py bm25search "action packed thriller" --limit 5
```

### Semantic Search
Embed movie data and search using meaning rather than exact keywords.
```bash
python cli/semantic_search_cli.py verify_embeddings
python cli/semantic_search_cli.py search "funny alien movie" --limit 5
```

### Hybrid Search
Perform search using combined strategies like RRF.
```bash
python cli/hybrid_search_cli.py rrf-search "time travel adventure" --limit 5
```

### Multimodal Search
Search the database using an image query.
```bash
python cli/multimodal_search_cli.py image_search data/sample_image.jpg --limit 5
```

### Retrieval Augmented Generation (RAG)
Generate conversational or summarized answers grounded in the search results.

**Standard RAG:**
```bash
python cli/augmented_generation_cli.py rag "What are some good sci-fi movies?"
```

**Summarized Output:**
```bash
python cli/augmented_generation_cli.py summarize "Show me options for kids movies"
```

**Citations-aware Answer:**
```bash
python cli/augmented_generation_cli.py citations "Documentaries about space"
```

### Evaluation
Evaluate the search engine using the provided golden dataset.
```bash
python cli/evaluation_cli.py --limit 5
```

## Technologies Used
- **Embeddings**: `sentence-transformers`
- **Generative AI**: Google GenAI API (`gemini-2.5-flash`, `gemma-4-31b-it`)
- **NLP**: `nltk`
- **Data & Vector Ops**: `numpy`
- **Image Processing**: `pillow`
