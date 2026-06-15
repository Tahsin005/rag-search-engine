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
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        indexes = self.index.get(term, set())
        return sorted(indexes)
    
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

def matches(query_tokens: list[str], title: str) -> bool:
    title_tokens = tokenize(title)

    return any(
        query_token == title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            print(f"Searching for: {preprocess(args.query)}")

            with open("data/movies.json", "r") as f:
                data = json.load(f)

            movies = data.get("movies", [])
            results = []

            query_tokens = tokenize(args.query)

            print(f"Query Tokens: {query_tokens}")

            for movie in movies:
                title = movie.get("title", "")

                if matches(query_tokens, title):
                    results.append(movie)

            print(f"Found {len(results)} movies:")

            for index, movie in enumerate(results[:5]):
                print(f"{index + 1}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()