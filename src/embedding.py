import os
import glob
import re
import json
from typing import List, Tuple
import numpy as np

import gestion_fichier as gf


from langchain_huggingface import HuggingFaceEmbeddings



from gestion_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()





from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 800 , chunk_overlap: int = 120 ) -> List[str]:
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


    def build(self, docs: List[Tuple[str, str]]):
        # Découpe chaque document en chunks
        all_chunks = []
        for doc in docs:
            # Récupère le texte et les métadonnées
            chunks = chunk_text(doc)
            self.save_ex_chunks(chunks)  # Sauvegarde les chunks extraits pour référence


            all_chunks.extend(chunks)
            # self.list_metadata(doc)
        print(f"[INFO] {len(all_chunks)} chunks créés à partir de {len(docs)} documents.")

        return all_chunks


    def save(self,all_chunks):
        # Sauvegarde l'index avec Chroma
        import chroma_database as chdt
        chro_db = chdt.ChromaDB(self.embedder)
        chro_db.save(all_chunks)
        self.vectordb = chro_db.vectordb


    def run(self,chunk_max_size : int = 800, chunk_overlap : int = 120):
        docs = gf.load_text_files()
        self.build(docs, chunk_max_size, chunk_overlap)
        return self.vectordb
    
    def telechargement(self):
        docs = gf.load_text_files()
        all_chunks = self.build(docs)
        self.save(all_chunks)


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


def call_app():
    # Test et utilisation du GPU si disponible
    device = test_GPU.test_utilisation_GPU()

    Embedding = Embedding_datasource(device)
    Embedding.run()
    print("\n ####### METADATA #######\n",Embedding.get_metadata())
    # print(Embedding.get_metadata())