import os
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = "cache/movie_embeddings.npy"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text in ["", None]:
            raise ValueError("Input text cannot be empty or None.")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        doc_strings = [f"{document['title']}: {document['description']}" for document in self.documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        os.makedirs("cache", exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        scored = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return results

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    sementic_search = SemanticSearch()
    embedding = sementic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def load_movies():
    import json
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data.get("movies", [])


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += step

    return chunks


def chunk_command(text, chunk_size=200, overlap=0):
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
