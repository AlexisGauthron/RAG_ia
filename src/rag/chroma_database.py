import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from langchain_community.vectorstores import Chroma

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()



class ChromaDB:
    def __init__(self, embedder, directory=f"{CHEMIN_FICHIER}/chroma_db"):
        self.directory = directory

        self.embedder = embedder
        self.vectordb = None

    def save(self,all_chunks):

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

    def delete(self):
        import shutil
        try:
            shutil.rmtree(self.directory)
            self.vectordb = None
            print(f"[INFO] Index supprimé de {self.directory}")
        except Exception as e:
            print(f"[ERROR] Erreur lors de la suppression de l'index: {e}")


    def get_chunks_db(self):
        if not self.vectordb:
            print("[INFO] Construction Index")
            self.vectordb = self.load()
        
         # Récupérer tous les documents, métadonnées et embeddings
        results = self.vectordb.similarity_search("", k=1000)
        
         # Transformer la liste d'objets Document en liste de dict {text, metadata}
        chunks = [{'text': doc.page_content, 'metadata': doc.metadata} for doc in results]

        print("CHUNKSSSSS :\n",chunks)
    
        return chunks
    

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
        import json
        import os
        if not self.vectordb:
            print("[INFO] Construction Index")
            self.vectordb = self.load()

        # Récupérer le chemin racine du projet en prenant le dossier parent du dossier du script
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'data'))
        output_dir = os.path.join(parent_dir, "all_chunks")
        os.makedirs(output_dir, exist_ok=True)

        # Nom du fichier json de sortie
        output_file = os.path.join(output_dir, "all_chunks.json")

        # Récupérer tous les documents (chunks) du vectordb - méthode depending du vectordb
        # Ici version générique pour des vecteurs Chroma
        all_docs = self.vectordb._collection.get()  # recupère tous les enregistrements dans une liste
        # Chaque doc contient typiquement 'documents' (chunk textuel) et 'metadatas'

        # Formater la sortie : liste de {chunk: texte, metadata: dict}
        data_to_export = []
        for doc in all_docs['documents']:
            index = all_docs['documents'].index(doc)
            chunk_text = doc
            metadata = all_docs['metadatas'][index] if 'metadatas' in all_docs and index < len(all_docs['metadatas']) else {}
            data_to_export.append({
                "chunk": chunk_text,
                "metadata": metadata
            })

        # Ecrire dans le fichier JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, ensure_ascii=False, indent=4)

        print(f"[INFO] Export des chunks réalisé dans : {output_file}")




    def compare_existe(self, directory):
        import os
        return os.path.exists(self.directory) and os.listdir(self.directory)
    

