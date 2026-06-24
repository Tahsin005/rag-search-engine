#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify CLIP model image embedding generation"
    )
    verify_parser.add_argument("image", type=str, help="Path to the image file to embed")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search the movie dataset using an image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to the image file for searching")
    image_search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
                print(f"   {result['description'][:100]}...\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
