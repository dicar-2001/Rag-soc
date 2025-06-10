import os
import re
from collections import Counter
import chainlit as cl
from processing import load_document, split_document, get_qwen3_embedding_model, embed_text
from vector_store import get_opensearch_client, index_chunks, search_vector, create_knn_index, bm25_rerank
from groq import Groq
# =====================
# MÉTRIQUES DÉTERMINISTES
# =====================
def deterministic_metrics(query, results_vector):
    """
    Calcule des métriques simples de pertinence entre la requête et les résultats.
    Retourne un dictionnaire avec : pertinence, chunk_relevance, most_common_term, matched_chunks.
    """
    if not results_vector:
        return {"pertinence": "Non", "chunk_relevance": "0.00", "most_common_term": "", "matched_chunks": 0}

    query_tokens = [w.strip(" ?.,;:!").lower() for w in query.split() if len(w) > 2]
    relevance_scores = []
    for chunk in results_vector:
        chunk_tokens = [w.strip(" ?.,;:!").lower() for w in chunk.split()]
        intersection = set(query_tokens).intersection(chunk_tokens)
        score = len(intersection) / len(query_tokens) if query_tokens else 0
        relevance_scores.append(score)

    matched_chunks = sum(1 for s in relevance_scores if s > 0)
    precision = matched_chunks / len(results_vector)
    top_chunk_tokens = set(results_vector[0].lower().split())
    pertinence = "Oui" if any(token in top_chunk_tokens for token in query_tokens) else "Non"
    all_text = " ".join(results_vector).lower()
    most_common_term = Counter(re.findall(r'\w+', all_text)).most_common(1)[0][0] if all_text else ""

    return {
        "pertinence": pertinence,
        "chunk_relevance": f"{precision:.2f}",
        "most_common_term": most_common_term,
        "matched_chunks": matched_chunks
    }

# =========================
# ÉVALUATION LLM AVEC GROQ API
# =========================
def llm_judge(question, answer):
    """
    Utilise le LLM Groq pour évaluer la réponse selon plusieurs critères RAG.
    Retourne une liste de Oui/Non pour chaque critère.
    """
    
    GROQ_API_KEY = "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"
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Tu es un évaluateur d'IA pour un système RAG.

Question : {question}
Réponse : {answer}

Pour chaque critère ci-dessous, réponds uniquement par Oui ou Non, séparés par des virgules, dans cet ordre :
1. context_sufficiency
2. correctness
3. relevance_to_query
4. groundedness
5. safety

Exemple de réponse attendue : Oui,Non,Oui,Oui,Oui

Donne seulement la liste, sans explication, sans phrase supplémentaire.
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_completion_tokens=30,
        top_p=1,
        stream=False,
    )
    generated_text = response.choices[0].message.content.strip()
    match = re.search(r"(oui|non)[^a-zA-Z]*(, *(oui|non)[^a-zA-Z]*){4}", generated_text, re.IGNORECASE)
    if match:
        fields = [x.strip().capitalize() for x in match.group(0).split(",")]
    else:
        fields = ["Non"] * 5
    return fields

# =========================
# PIPELINE PRINCIPAL + CHAINLIT
# =========================
# Initialisation globale (avant la fonction on_message)
docs_dir = "docs"
supported_ext = ('.txt', '.pdf', '.docx')
all_chunks = []
for filename in os.listdir(docs_dir):
    file_path = os.path.join(docs_dir, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(supported_ext):
        text = load_document(file_path)
        if text.strip():
            chunks = split_document(text)
            all_chunks.extend(chunks)

tokenizer, model = get_qwen3_embedding_model()
embeddings = [embed_text(chunk, tokenizer, model) for chunk in all_chunks]
client = get_opensearch_client()
index_name = "rag-index"
create_knn_index(index_name, client, dim=embeddings[0].shape[0])
index_chunks(all_chunks, embeddings, index_name, client)

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content
    query_embedding = embed_text(query, tokenizer, model)
    results = search_vector(query_embedding, index_name, client)
    if results["hits"]["hits"]:
        top_hits = results["hits"]["hits"][:5]
        results_vector = [hit["_source"]["content"] for hit in top_hits]
        reranked_chunks = bm25_rerank(query, results_vector, top_n=3)
        fused_context = "\n".join(reranked_chunks)
        det_metrics = deterministic_metrics(query, results_vector)
        llm_fields = llm_judge(query, fused_context)
        response = (
            f"**Réponse de retrieval (fusion top 3) :**\n{fused_context}\n\n"
            f"**--- Métriques déterministes ---**\n"
            f"Pertinence : {det_metrics['pertinence']}\n"
            f"Chunk relevance/precision : {det_metrics['chunk_relevance']}\n"
            f"Mot le plus fréquent : {det_metrics['most_common_term']}\n"
            f"Chunks pertinents : {det_metrics['matched_chunks']}\n\n"
            f"**--- Évaluation LLM ---**\n"
            f"context_sufficiency : {llm_fields[0]}\n"
            f"correctness : {llm_fields[1]}\n"
            f"relevance_to_query : {llm_fields[2]}\n"
            f"groundedness : {llm_fields[3]}\n"
            f"safety : {llm_fields[4]}"
        )
    else:
        response = "Aucun résultat trouvé."
    await cl.Message(content=response).send()







