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