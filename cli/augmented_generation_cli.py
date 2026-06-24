#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch

from dotenv import load_dotenv
from google import genai
from google.genai import types

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform multi-document summarization"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summary")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of search results to summarize")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            
            documents = load_movies()
            hybrid = HybridSearch(documents)
            
            # perform RRF search for top 5 results
            results = hybrid.rrf_search(query, k=60, limit=5)
            
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            
            client = genai.Client(api_key=api_key)
            
            formatted_docs = []
            for doc in results:
                formatted_docs.append(f"Title: {doc.get('title', '')}\nDescription: {doc.get('document', '')}")
            
            docs_str = "\n\n".join(formatted_docs)
            
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs_str}

Answer:"""

            try:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                )
                
                print("Search Results:")
                for doc in results:
                    print(f"- {doc['title']}")
                    
                print("\nRAG Response:")
                if response.text:
                    print(response.text.strip())
                else:
                    print("Empty response from the model.")
                    
            except Exception as e:
                print(f"RAG generation failed: {e}")

        case "summarize":
            query = args.query
            limit = args.limit
            
            documents = load_movies()
            hybrid = HybridSearch(documents)
            
            results = hybrid.rrf_search(query, k=60, limit=limit)
            
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            
            client = genai.Client(api_key=api_key)
            
            formatted_docs = []
            for doc in results:
                formatted_docs.append(f"Title: {doc.get('title', '')}\nDescription: {doc.get('document', '')}")
            
            docs_str = "\n\n".join(formatted_docs)
            
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{docs_str}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            try:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                )
                
                print("Search Results:")
                for doc in results:
                    print(f"  - {doc['title']}")
                    
                print("\nLLM Summary:")
                if response.text:
                    print(response.text.strip())
                else:
                    print("Empty response from the model.")
                    
            except Exception as e:
                print(f"Summarization failed: {e}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
