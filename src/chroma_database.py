from langchain_community.vectorstores import Chroma

from load_fichier import chemindossier
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


    def compare_existe(self):
        import os
        return os.path.exists(self.directory) and os.listdir(self.directory)
    

