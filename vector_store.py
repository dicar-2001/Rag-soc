from opensearchpy import OpenSearch
from typing import List
from rank_bm25 import BM25Okapi


def get_opensearch_client(host="localhost", port=9200) -> OpenSearch:
    """Crée un client OpenSearch."""
    return OpenSearch([{'host': host, 'port': port}])

def create_knn_index(index_name, client, dim=1024):
    """Crée un index OpenSearch avec un champ knn_vector pour les embeddings."""
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim
                }
            }
        }
    }
    client.indices.create(index=index_name, body=body)

def index_chunks(chunks: List[str], embeddings: List, index_name: str, client):
    """Indexe les chunks et leurs embeddings dans OpenSearch."""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {"content": chunk, "embedding": embedding.tolist()}
        client.index(index=index_name, id=i, body=doc)

def search_vector(query_embedding, index_name: str, client, top_k: int = 5):
    """Recherche les chunks les plus similaires à l'embedding de la requête."""
    body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        }
    }
    return client.search(index=index_name, body=body)

def bm25_rerank(query: str, chunks: list, top_n: int = 3) -> list:
    """Rerank les chunks avec BM25 et retourne les top_n plus pertinents."""
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_n]]


