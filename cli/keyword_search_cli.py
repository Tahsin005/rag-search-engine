#!/usr/bin/env python3

import argparse
import json
import string

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # preprocessing the query
            translation_table = str.maketrans("", "", string.punctuation)
            query = args.query.lower().translate(translation_table)

            print(f"Searching for: {query}")

            # loading the data
            data = json.load(open("data/movies.json", "r"))
            movies = data.get("movies", [])

            results = []

            for movie in movies:
                title = movie.get("title", "").lower().translate(translation_table)

                if query in title:
                    results.append(movie)

            print(f"Found {len(results)} movies:")
            for index, movie in enumerate(results[:5]):
                print(f"{index + 1}. {movie['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()