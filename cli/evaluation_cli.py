#!/usr/bin/env python3

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    hybrid = HybridSearch(documents)

    print(f"k={limit}\n")

    for case in golden_dataset["test_cases"]:
        query = case["query"]
        relevant_docs = case["relevant_docs"]

        results = hybrid.rrf_search(query, k=60, limit=limit)
        
        retrieved_titles = [res["title"] for res in results]
        
        relevant_retrieved = len(set(retrieved_titles) & set(relevant_docs))
        total_retrieved = len(retrieved_titles)
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}\n")


if __name__ == "__main__":
    main()
