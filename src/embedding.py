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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline


# Tout les loadeurs de documents
from langchain.document_loaders import (
    TextLoader,            # Pour .txt
    UnstructuredMarkdownLoader,  # Pour .md
    PythonLoader,          # Pour .py
    JSONLoader,            # Pour .json
    PyPDFLoader,           # Pour .pdf
    UnstructuredWordDocumentLoader,  # Pour .docx
    UnstructuredExcelLoader,         # Pour .xlsx
    UnstructuredPowerPointLoader,    # Pour .pptx
    UnstructuredHTMLLoader,          # Pour .html/.htm
    UnstructuredEmailLoader,         # Pour .eml
    CSVLoader,             # Pour .csv
    UnstructuredFileLoader # Pour fichiers divers (générique)
)

# Types de fichiers supportés
LISTE_FICHIER_ACCEPTE = [".txt", ".md", ".py", ".json", ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm", ".eml", ".csv"]

EXTENSION_LOADER_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": PythonLoader,
    ".json": JSONLoader,
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".eml": UnstructuredEmailLoader,
    ".csv": CSVLoader,
}


from path_file import chemindossier
CHEMIN_FICHIER = chemindossier()



# -----------------------------
# 1) Chargement & préparation
# -----------------------------
def load_text_files(data_dir: str = f"{CHEMIN_FICHIER}/data_rag") -> List[Tuple[str, str]]:
    """
    Charge tous les fichiers .txt/.md/.py d'un dossier.
    Retourne une liste [(path, text), ...]
    """
    
    # Récupère tous les fichiers (récursivement)
    all_files = glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)

    # Filtre selon l'extension
    paths = []
    if not all_files:
        print(f"[WARN] Aucun fichier trouvé dans le répertoire {data_dir}. \nVeuillez vérifier le chemin : {os.path.abspath(data_dir)}")
    else:
        for f in all_files:
            if os.path.splitext(f)[1] in LISTE_FICHIER_ACCEPTE:
                paths.append(f)
                print("[INFO] Fichier trouvé:", f)
            else:
                print("[WARN] Fichier ignoré (extension non supportée):", f)

    docs = []
    
    for p in paths:
        doc = []
        # Choix du loader selon l'extension
        extension = os.path.splitext(p)[1]
        loader_cls = EXTENSION_LOADER_MAP.get(extension)
        if loader_cls:
            loader = loader_cls(p)
            
        try:
            doc = loader.load()
        except Exception as e:
            print(f"[WARN] Impossible de lire {p}: {e}")
        # print(doc,"\n\n")
        docs.append(doc)

    return docs



from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int , chunk_overlap: int ) -> List[str]:
    """
    Découpe un texte en morceaux qui se recouvrent légèrement.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,         # longueur max de chaque segment (en caractères)
        chunk_overlap=chunk_overlap,       # chevauchement entre les segments
        separators=["\n\n", "\n", ".", " "]  # ordre de priorité pour les coupures
    )
    chunks = splitter.split_documents(text)
    # print("\n\n",chunks,"\n\n")

    return chunks




# -----------------------------
# 2) Index FAISS (création / chargement)
# -----------------------------
class Embedding_datasource:
    def __init__(self, device, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": device}
        ) 
        self.vectordb = None
        self.metadata = set()

    def build(self, docs: List[Tuple[str, str]], chunk_max_size, chunk_overlap):
        # Découpe chaque document en chunks
        all_chunks = []
        for doc in docs:
            # Récupère le texte et les métadonnées
            chunks = chunk_text(doc, chunk_max_size, chunk_overlap)
            self.save_ex_chunks(chunks)  # Sauvegarde les chunks extraits pour référence


            all_chunks.extend(chunks)
            # self.list_metadata(doc)
        print(f"[INFO] {len(all_chunks)} chunks créés à partir de {len(docs)} documents.")


        # Sauvegarde l'index avec Chroma
        import chroma_database as chdt
        chro_db = chdt.ChromaDB(self.embedder)
        chro_db.save(all_chunks)
        self.vectordb = chro_db.vectordb


    def run(self,chunk_max_size : int = 800, chunk_overlap : int = 120):
        docs = load_text_files()
        self.build(docs, chunk_max_size, chunk_overlap)
        return self.vectordb
    


    def save_ex_chunks(self, chunks):
        """
        Sauvegarde uniquement les 2 premiers chunks dans un fichier JSON,
        nommé d'après le champ metadata['source'] du premier chunk.
        Crée automatiquement le dossier chroma_res/ si nécessaire.
        """
        try:
            # ✅ Vérification de la présence de chunks
            if not chunks:
                print("[WARN] Aucun chunk à sauvegarder.")
                return

            # ✅ Création du dossier s'il n'existe pas
            path_dos = os.path.dirname(f"{CHEMIN_FICHIER}/chroma_res/")
            os.makedirs(path_dos, exist_ok=True)

            # ✅ Récupération de la source depuis le metadata du premier chunk
            source_path = chunks[0].metadata.get("source", "unknown_source")

            # Extraction et nettoyage du nom du fichier
            base_name = os.path.basename(source_path)
            file_name = os.path.splitext(base_name)[0]
            file_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', file_name)

            # ✅ Définir le chemin final du fichier JSON
            filename = f"{path_dos}/{file_name}_chunks.json"

            # ✅ Limiter à 2 chunks
            chunks_to_save = chunks[:2]

            # ✅ Sauvegarde des chunks dans le fichier
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            # "page_content": c.page_content,
                            "metadata": c.metadata
                        } for c in chunks_to_save
                    ],
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            print(f"[INFO] {len(chunks_to_save)} chunks sauvegardés dans {filename}")

        except Exception as e:
            print(f"[ERROR] Erreur lors de la sauvegarde des chunks: {e}")









    def list_metadata(self, docs):
        """Retourne toutes les clés de métadonnées distinctes présentes dans docs."""

        def walk(item):
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    walk(sub)
                return
            if hasattr(item, "metadata"):
                md = getattr(item, "metadata", {}) or {}
                if isinstance(md, dict):
                    self.metadata.update(md.keys())

        walk(docs)


    def get_metadata(self):
        return sorted(self.metadata)
    



####### Lancement de tests #######

import utilisation_GPU as test_GPU


if __name__ == "__main__":
    # Test et utilisation du GPU si disponible
    device = test_GPU.test_utilisation_GPU()

    Embedding = Embedding_datasource(device)
    Embedding.run()
    print("\n ####### METADATA #######\n",Embedding.get_metadata())
    # print(Embedding.get_metadata())

