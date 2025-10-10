import os
import glob
import re
import json
from typing import List, Tuple
import numpy as np

# Embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

# LLM (petit modèle CPU-friendly)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# -----------------------------
# 1) Chargement & préparation
# -----------------------------
def load_text_files(data_dir: str = "data") -> List[Tuple[str, str]]:
    """
    Charge tous les fichiers .txt/.md/.py d'un dossier.
    Retourne une liste [(path, text), ...]
    """
    paths = []
    for ext in ("*.txt", "*.md", "*.py"):
        paths += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)

    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            # Un petit nettoyage
            txt = re.sub(r"\s+\n", "\n", txt).strip()
            if txt:
                docs.append((p, txt))
        except Exception as e:
            print(f"[WARN] Impossible de lire {p}: {e}")
    return docs


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """
    Découpe un texte en morceaux qui se recouvrent légèrement.
    """
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - chunk_overlap)
    return chunks


# -----------------------------
# 2) Index FAISS (création / chargement)
# -----------------------------
class VectorIndex:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.metadatas = []  # [(path, idx_chunk, text)]
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def build(self, docs: List[Tuple[str, str]], chunk_size=800, chunk_overlap=120):
        """
        Construit l'index FAISS à partir des documents.
        """
        chunks = []
        metas = []
        for path, text in docs:
            for i, ch in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
                chunks.append(ch)
                metas.append((path, i, ch))

        print(f"[INFO] Nombre de chunks: {len(chunks)}")
        embs = self.embedder.encode(chunks, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

        self.index = faiss.IndexFlatIP(self.dim)  # IP = inner product (avec vecteurs normalisés => cos sim)
        self.index.add(embs)
        self.metadatas = metas

    def save(self, out_dir: str = "index"):
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "faiss.index"))
        # Sauvegarde des métadonn
