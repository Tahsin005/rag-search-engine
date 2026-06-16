#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string
from collections import Counter
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    return text.lower().translate(TRANSLATION_TABLE)


def load_stopwords() -> set[str]:
    with open("data/stopwords.txt", "r") as f:
        return {preprocess(word) for word in f.read().splitlines()}


STOP_WORDS = load_stopwords()


def tokenize(text: str) -> list[str]:
    tokens = preprocess(text).split()
    return [
        stemmer.stem(token)
        for token in tokens
        if token and token not in STOP_WORDS
    ]

def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token from '{term}', got {len(tokens)}")
    return tokens[0]

def load_movies() -> list:
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data.get("movies", [])


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokenize(text):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        for path in ["cache/index.pkl", "cache/docmap.pkl", "cache/term_frequencies.pkl"]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file '{path}' not found. Run 'build' first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            search_command(args.query)
        case "tf":
            tf_command(args.doc_id, args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()