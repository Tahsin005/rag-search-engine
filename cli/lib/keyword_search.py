import os
import pickle
import math
from collections import Counter

from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    tokenize,
    tokenize_term,
    load_movies,
)


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str):
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        term_doc_count = len(self.index.get(term, set()))
        return math.log((total_docs + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        term_doc_count = len(self.index.get(term, set()))
        return math.log((total_docs - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        query_tokens = tokenize(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            total = 0.0
            for token in query_tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        return [(self.docmap[doc_id], scores[doc_id]) for doc_id in sorted_ids[:limit]]

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        paths = [self.index_path, self.docmap_path, self.tf_path, self.doc_lengths_path]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file '{path}' not found. Run 'build' first.")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    query_tokens = tokenize(query)
    print(f"Query Tokens: {query_tokens}")

    results = []
    seen_ids = set()

    for token in query_tokens:
        for doc_id in idx.get_documents(token):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(idx.docmap[doc_id])
            if len(results) >= 5:
                break
        if len(results) >= 5:
            break

    print(f"Found {len(results)} movies:")
    for i, movie in enumerate(results):
        print(f"{i + 1}. {movie['title']} (id: {movie['id']})")


def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    print(idx.get_tf(doc_id, token))


def idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    idf = idx.get_idf(token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def bm25_idf_command(term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    bm25idf = idx.get_bm25_idf(token)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0
    token = tokenize_term(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)


def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    tfidf = idx.get_tf(doc_id, token) * idx.get_idf(token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")


def bm25_search_command(query: str, limit: int = 5):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = idx.bm25_search(query, limit)
    for i, (movie, score) in enumerate(results):
        print(f"{i + 1}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")