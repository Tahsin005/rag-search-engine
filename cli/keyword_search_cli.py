#!/usr/bin/env python3

import argparse
import json
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


def matches(query_tokens: list[str], title: str) -> bool:
    title_tokens = tokenize(title)

    return any(
        query_token == title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

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