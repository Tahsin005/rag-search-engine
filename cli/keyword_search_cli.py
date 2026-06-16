#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string
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


def load_movies() -> list:
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data.get("movies", [])


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str):
        for token in tokenize(text):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

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

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Index files not found. Run 'build' first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            search_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()