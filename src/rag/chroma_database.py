import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from langchain_community.vectorstores import Chroma
from langchain.schema import Document  # ou langchain.docstore.document import Document

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()
CHEMIN_FICHIER_RAG = f"{chemindossier()}/data_rag"

import src.rag.embedding as emb


def dicts_to_documents(chunks: list[dict]) -> list[Document]:
    documents = []
    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}
        documents.append(Document(page_content=text, metadata=metadata))
    print("[INFO] Changement dicts to document")
    return documents

def is_list_of_dicts(x, *, allow_empty=True) -> bool:
    if not isinstance(x, list):
        return False
    if not x:       # liste vide
        return allow_empty
    return all(isinstance(item, dict) for item in x)

# Transformer la liste d'objets Document en liste de dict {text, metadata}
def documents_to_dict(results: list[Document]) -> list[dict]:
    return [{'text': doc.page_content, 'metadata': doc.metadata} for doc in results]




class ChromaDB:
    def __init__(self, embedder, directory=f"{CHEMIN_FICHIER}/chroma_db"):
        self.directory = directory

        self.embedder = embedder
        self.vectordb = None

    def save(self,all_chunks):
        
        # Test si all chunks est une liste de dictionnaire
        if is_list_of_dicts(all_chunks):
            all_chunks = dicts_to_documents(all_chunks)


        # Indexation avec Chroma
        self.vectordb = Chroma.from_documents(all_chunks, self.embedder, persist_directory=self.directory)
        
        if self.vectordb:
            # self.vectordb.persist()
            print(f"[INFO] Index sauvegardé")
        else:
            print("[WARN] Aucun index à sauvegarder.")
    

    def load(self):
        try:
            self.vectordb = Chroma(persist_directory=self.directory, embedding_function=self.embedder)
            print(f"[INFO] Index chargé depuis {self.directory}")
        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement de l'index: {e}")
            self.vectordb = None
        return self.vectordb


    def delete_files(self, nom_fichier: str, check_doublons: bool = False):
        """
        Supprime tous les chunks dont la métadonnée `source`
        correspond au nom de fichier fourni.
        """
        print(f"[INFO] Section Deletes files {nom_fichier} !\n")

        if not nom_fichier:
            print("[WARN] Aucun nom de fichier fourni, suppression annulée.\n")
            return

        if not self.vectordb:
            self.vectordb = self.load()

        if not self.vectordb:
            print("[ERROR] Impossible de charger l'index, suppression annulée.\n")
            return

        try:
            ids = self.affichage_match(nom_fichier)

            if ids:
                self.vectordb.delete(ids=ids)
                print(f"[INFO] {len(ids)} chunk(s) supprimé(s) pour source='{nom_fichier}'.")
                if check_doublons:
                    print(f"[INFO] Suppression doublons database!\n")
                return 1
            else:
                print("[INFO] Aucun chunk à supprimer pour cette source.")
                if check_doublons:
                    print(f"[INFO] Pas de doublons dans la database!\n")
                return 0


                

        except Exception as e:
            print(f"[ERROR] Erreur lors de la suppression des chunks:\n {e}")



    def affichage_match(self, nom_fichier: str):
        print("[INFO] Affichage match source !\n")

        if not self.vectordb:
            self.vectordb = self.load()
        if not self.vectordb:
            print("[ERROR] Impossible de charger l’index.\n")
            return []

        collection = getattr(self.vectordb, "_collection", self.vectordb)

        # Recuperer tous les documents et filtrer par basename,
        # car le champ "source" stocke le chemin complet du fichier
        # alors que nom_fichier est un basename (ex: "document.pdf")
        all_data = collection.get(include=["metadatas"])
        all_ids = all_data.get("ids", [])
        all_metas = all_data.get("metadatas", [])

        # Normaliser les separateurs (\ Windows et / Unix) avant basename
        ids = [
            doc_id
            for doc_id, meta in zip(all_ids, all_metas)
            if os.path.basename(meta.get("source", "").replace("\\", "/")) == nom_fichier
        ]

        print("[INFO] Nombre de matches :", len(ids), "\n")
        return ids


    def delete_all(self):
        """
        Supprime complètement la base Chroma persistée.
        """
        import shutil
        try:
            shutil.rmtree(self.directory)
            self.vectordb = None
            print(f"[INFO] Base Chroma supprimée: {self.directory}")
        except FileNotFoundError:
            print(f"[INFO] Aucun index à supprimer dans {self.directory}")
        except Exception as e:
            print(f"[ERROR] Erreur lors de la suppression de la base: {e}")


    def get_chunks_db(self):
        """Récupère TOUS les chunks de la base sans limite artificielle."""
        if not self.vectordb:
            print("[INFO] Construction Index")
            self.vectordb = self.load()

        # Utiliser l'API Chroma directement pour récupérer TOUS les documents
        # (pas de limite de 1000 comme avec similarity_search)
        all_data = self.vectordb._collection.get(include=["documents", "metadatas"])

        chunks = []
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunks.append({"text": doc, "metadata": metadata})

        return chunks
    


    def mise_a_jour_metadata(self):
        """Met a jour les metadonnees en place sans supprimer/recreer la base."""
        if not self.vectordb:
            self.load()
        if not self.vectordb:
            return

        collection = self.vectordb._collection
        all_data = collection.get(include=["documents", "metadatas"])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])

        if not ids:
            print("[INFO] Aucun chunk a mettre a jour.")
            return

        # Construire les chunks pour augmentation_metadonne
        chunks = [{"text": documents[i], "metadata": metadatas[i]} for i in range(len(ids))]
        chunks = emb.augmentation_metadonne(chunks)

        # Mettre a jour les metadonnees en place
        updated_metadatas = [c["metadata"] for c in chunks]
        collection.update(ids=ids, metadatas=updated_metadatas)
        print(f"[INFO] Metadonnees mises a jour pour {len(ids)} chunks (in-place)")



    def all_metadata(self) -> list:
        """
        Récupère toutes les métadonnées stockées dans la collection Chroma.

        Returns:
            list: Liste des dictionnaires de métadonnées pour chaque document.
        """
        if not self.vectordb:
            print("[INFO] Index non chargé, chargement en cours...")
            self.vectordb = self.load()  # ou équivalent

        # Récupère tous les métadonnées de la collection (sans limiter)
        results = self.vectordb._collection.get(include=["metadatas"])

        metadatas = results.get("metadatas", [])

        # Extraire toutes les clés uniques dans toutes les métadonnées
        keys = set()
        for meta in metadatas:
            keys.update(meta.keys())

        print(f"\nMetadata unique : {keys}\n")

        return keys
    


        

    

    def write_all_chunks(self):
        """Exporte tous les chunks dans un fichier JSON."""
        import json

        print("[INFO] Export des chunks...")

        if not self.vectordb:
            self.vectordb = self.load()

        # Chemin de sortie
        output_dir = f"{CHEMIN_FICHIER}/all_chunks"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "all_chunks.json")

        # Récupérer tous les documents via l'API Chroma
        all_docs = self.vectordb._collection.get(include=["documents", "metadatas"])

        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        # Algorithme O(n) : utiliser enumerate au lieu de .index()
        data_to_export = [
            {
                "chunk": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {}
            }
            for i, doc in enumerate(documents)
        ]

        # Ecrire dans le fichier JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, ensure_ascii=False, indent=4)

        print(f"[INFO] Export des chunks réalisé dans : {output_file}")




    def overwrite_db(self, all_chunks):
        self.delete_all()
        self.save(all_chunks)
    

