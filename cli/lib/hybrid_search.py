import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

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


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

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