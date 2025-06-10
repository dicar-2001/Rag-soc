from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

def load_document(file_path: str) -> str:
    """Charge le contenu d'un document texte, PDF ou Word."""
    ext = file_path.lower().split('.')[-1]
    if ext == "txt":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    elif ext == "pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""  # Ignore unsupported files

def split_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Découpe le texte en chunks avec chevauchement."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_qwen3_embedding_model(model_name="Qwen/Qwen3-Embedding-0.6B"):
    """Charge le tokenizer et le modèle Qwen3 Embedding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model

def embed_text(text, tokenizer, model):
    """Génère l'embedding d'un texte avec Qwen3 Embedding."""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].cpu().numpy()