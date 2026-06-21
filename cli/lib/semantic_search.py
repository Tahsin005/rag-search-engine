import os
import re
import json
from typing import TypedDict
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"
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


# semantic chunking
def semantic_chunk_text(text, max_chunk_size=4, overlap=0):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than max chunk size.")

    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + max_chunk_size])
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if i + max_chunk_size >= len(sentences):
            break
        i += step

    return chunks


def semantic_chunk_command(text, max_chunk_size=4, overlap=0):
    chunks = semantic_chunk_text(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")



class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata: list[ChunkMetadata] = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description or not description.strip():
                continue

            chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[i]
            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": score,
            })

        movie_scores = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in movie_scores or cs["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = cs

        sorted_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for entry in top_movies:
            doc = self.documents[entry["movie_idx"]]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc.get("description", "")[:100],
                "score": round(float(entry["score"]), 4),
                "metadata": {"chunk_idx": entry["chunk_idx"]},
            })

        return results


def embed_chunks_command():
    documents = load_movies()
    chunked_search = ChunkedSemanticSearch()
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int = 5):
    documents = load_movies()
    chunked_search = ChunkedSemanticSearch()
    chunked_search.load_or_create_chunk_embeddings(documents)
    results = chunked_search.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")