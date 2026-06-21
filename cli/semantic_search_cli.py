#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
    load_movies,
)


def search_command(query, limit):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:1000]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads correctly")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text input")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Build or load movie embeddings and verify them")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a query string")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()