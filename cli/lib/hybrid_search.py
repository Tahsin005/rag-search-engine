import os
import time
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        results = self.idx.bm25_search(query, limit)
        return [
            {"id": doc["id"], "title": doc["title"], "score": score}
            for doc, score in results
        ]

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = [r["score"] for r in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]

        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        combined: dict[int, dict] = {}

        for result, norm_score in zip(bm25_results, normalized_bm25):
            doc_id = result["id"]
            combined[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_score": norm_score,
                "semantic_score": 0.0,
            }

        for result, norm_score in zip(semantic_results, normalized_semantic):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "document": self.idx.docmap.get(doc_id, result),
                    "bm25_score": 0.0,
                    "semantic_score": norm_score,
                }
            else:
                combined[doc_id]["semantic_score"] = norm_score

        results = []
        for doc_id, entry in combined.items():
            score = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)
            results.append({
                "id": doc_id,
                "title": entry["document"]["title"],
                "document": entry["document"].get("description", ""),
                "bm25_score": entry["bm25_score"],
                "semantic_score": entry["semantic_score"],
                "hybrid_score": score,
            })

        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return results


    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined: dict[int, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            combined[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            score = rrf_score(rank, k)
            if doc_id not in combined:
                combined[doc_id] = {
                    "document": self.idx.docmap.get(doc_id, result),
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": score,
                }
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf_score"] += score

        results = []
        for doc_id, entry in combined.items():
            results.append({
                "id": doc_id,
                "title": entry["document"]["title"],
                "document": entry["document"].get("description", ""),
                "bm25_rank": entry["bm25_rank"],
                "semantic_rank": entry["semantic_rank"],
                "rrf_score": entry["rrf_score"],
            })

        results.sort(key=lambda r: r["rrf_score"], reverse=True)
        return results[:limit]
        

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_command(scores: list[float]):
    normalized = normalize_scores(scores)
    for score in normalized:
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5):
    documents = load_movies()
    hybrid = HybridSearch(documents)
    results = hybrid.weighted_search(query, alpha, limit)

    for i, result in enumerate(results[:limit], start=1):
        print(f"{i}. {result['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
        print(f"  {result['document'][:100]}...")


def rrf_search_command(query: str, k: int = 60, limit: int = 5, enhance: str = None, rerank_method: str = None, debug: bool = False, evaluate: bool = False):
    logger = logging.getLogger(__name__)
    original_query = query
    logger.debug("[1/4] Original query: '%s'", original_query)
    if enhance in ["spell", "rewrite", "expand"]:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        
        if enhance == "spell":
            prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        elif enhance == "rewrite":
            prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        elif enhance == "expand":
            prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt
        )
        query = response.text.strip()
        logger.debug("[2/4] Query after '%s' enhancement: '%s'", enhance, query)
        print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")

    documents = load_movies()
    hybrid = HybridSearch(documents)
    
    fetch_limit = limit * 5 if rerank_method in ["individual", "batch", "cross_encoder"] else limit
    results = hybrid.rrf_search(query, k, fetch_limit)
    logger.debug(
        "[3/4] RRF search returned %d results: %s",
        len(results),
        [r["title"] for r in results],
    )

    if rerank_method == "individual":
        print(f"Re-ranking top {len(results)} results using {rerank_method} method...")
        
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        
        for doc in results:
            prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
            try:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    )
                )
                score_str = response.text.strip()
                doc["rerank_score"] = float(score_str)
            except Exception as e:
                doc["rerank_score"] = 0.0
            time.sleep(3)
            
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    elif rerank_method == "batch":
        print(f"Re-ranking top {len(results)} results using {rerank_method} method...")
        
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        
        doc_list_str = ""
        for doc in results:
            doc_list_str += f"ID: {doc['id']}\nTitle: {doc.get('title', '')}\nDescription: {doc.get('document', '')}\n\n"
            
        prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
            )
            ranked_ids = json.loads(response.text.strip())
            rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
            for doc in results:
                doc["rerank_rank"] = rank_map.get(doc["id"], float("inf"))
            
            results.sort(key=lambda x: x.get("rerank_rank", float("inf")))
        except Exception as e:
            print(f"Batch reranking failed: {e}")
    elif rerank_method == "cross_encoder":
        print(f"Re-ranking top {len(results)} results using {rerank_method} method...")
        
        pairs = []
        for doc in results:
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)
        
        for doc, score in zip(results, scores):
            doc["cross_encoder_score"] = float(score)
            
        results.sort(key=lambda x: x.get("cross_encoder_score", float("-inf")), reverse=True)

    logger.debug(
        "[4/4] Final results after re-ranking (top %d): %s",
        limit,
        [r["title"] for r in results[:limit]],
    )

    print(f"Reciprocal Rank Fusion Results for '{original_query}' (k={k}):\n")

    for i, result in enumerate(results[:limit], start=1):
        bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "-"
        semantic_rank = result["semantic_rank"] if result["semantic_rank"] is not None else "-"
        print(f"{i}. {result['title']}")
        if "rerank_score" in result:
            print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
        elif "rerank_rank" in result and result["rerank_rank"] != float("inf"):
            print(f"   Re-rank Rank: {result['rerank_rank']}")
        elif "cross_encoder_score" in result:
            print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
        print(f"   {result['document'][:100]}...")
        print()

    if evaluate:
        print("Evaluating results...")
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        
        formatted_results = []
        for doc in results[:limit]:
            formatted_results.append(f"{doc.get('title', '')} - {doc.get('document', '')}")
            
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

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
            
            if not response.text:
                print("Evaluation failed: The model returned an empty response (likely blocked by safety filters).")
                if response.candidates and response.candidates[0].finish_reason:
                    print(f"Finish reason: {response.candidates[0].finish_reason}")
                return
                
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            scores = json.loads(text)
            
            for i, (doc, score) in enumerate(zip(results[:limit], scores), start=1):
                print(f"{i}. {doc['title']}: {score}/3")
        except Exception as e:
            print(f"Evaluation failed: {e}")