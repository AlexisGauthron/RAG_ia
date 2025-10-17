from __future__ import annotations
from typing import List, Union
import re
import unicodedata

# Compat imports (selon ta version LangChain)
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Nettoyage léger, safe pour l'embedding ---
def clean_text(s: str) -> str:
    # Normalisation unicode
    s = unicodedata.normalize("NFKC", s)

    # Supprimer espaces en fin de ligne
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)

    # Déhyphénation des césures en fin de ligne: "infor-\nmation" -> "information"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # Remplacer sauts de ligne multiples par au plus 2 (préserve paragraphes)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Collapser espaces multiples
    s = re.sub(r"[ \t]{2,}", " ", s)

    # Lignes de numéros de page / en-têtes très simples (heuristique)
    s = re.sub(r"(?m)^(page|p\.?)\s*\d+\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?m)^\s*Table of Contents\s*$", "", s, flags=re.IGNORECASE)

    # URLs trop verbeuses -> les garder peut nuire au signal; on les remplace par un jeton
    s = re.sub(r"https?://\S+", "<URL>", s)

    # Trim global
    return s.strip()


def clean_document(doc: Document) -> Document:
    return Document(
        page_content=clean_text(doc.page_content or ""),
        metadata=dict(doc.metadata or {})
    )