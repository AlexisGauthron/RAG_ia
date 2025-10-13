from langchain_community.vectorstores import Chroma

from path_file import chemindossier
CHEMIN_FICHIER = chemindossier()



class ChromaDB:
    def __init__(self, embedder, directory=f"{CHEMIN_FICHIER}/chroma_db"):
        self.directory = directory
        self.embedder = embedder
        self.vectordb = None

    def query(self, query_text, top_k=5, metadata_filter=None):
        try:
            results = self.vectordb.similarity_search(
                query_text, k=top_k, filter=metadata_filter
            )
            return results
        except Exception as e:
            print(f"[ERROR] Erreur lors de la requête: {e}")
            return []
    

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


    def delete(self):
        import shutil
        try:
            shutil.rmtree(self.directory)
            self.vectordb = None
            print(f"[INFO] Index supprimé de {self.directory}")
        except Exception as e:
            print(f"[ERROR] Erreur lors de la suppression de l'index: {e}")

    

