from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lib.search_utils import load_movies

class MultimodalSearch:
    def __init__(self, documents=None, model_name: str = "clip-ViT-B-32"):
        self.documents = documents or []
        self.model = SentenceTransformer(model_name)
        
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_tensor=True)
        else:
            self.text_embeddings = None

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        # pass a one-element list to encode and take the first/only element
        embedding = self.model.encode([image], convert_to_tensor=True)[0]
        return embedding

    def search_with_image(self, image_path: str, limit: int = 5):
        if self.text_embeddings is None or not self.documents:
            return []
            
        image_embedding = self.embed_image(image_path)
        similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]
        
        results = []
        for i, doc in enumerate(self.documents):
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "similarity": similarities[i].item()
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str, limit: int = 5):
    documents = load_movies()
    searcher = MultimodalSearch(documents)
    results = searcher.search_with_image(image_path, limit)
    return results
