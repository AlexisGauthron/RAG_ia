
# Fichier inclue dans module_app.py
import test.utilisation_GPU as test_GPU
import src.rag.load_fichier as gf
import src.rag.embedding as emb
import src.rag.chroma_database as chdt
import src.rag.rag as rg

import src.modele.modele_Embeddings as modele_Emb
import src.modele.modele_LLM_ollama as mode_oll

import src.rag.prompt as prompt

class module_app:
    def __init__(self, embed_model, prompt_model: int, directory: str = gf.chemindossier()):
        self.device = test_GPU.test_utilisation_GPU()
        self.embed_model = modele_Emb.Model_embeddings(self.device,embed_model).get_embedder()
        self.prompt_model = prompt_model

         # Initialisation des composants RAG

        self.embedder = emb.Embedding_datasource(self.embed_model)
        self.chro_db = chdt.ChromaDB(self.embed_model)
        self.rag = None



    # Fonction pour créer les embeddings et la base vectorielle
    def telechargement(self,data_folder: str):
        
        docs = gf.load_text_files(data_folder)
        all_chunks = self.embedder.build_all_chunks(docs)
        # Augmentation des métas données
        all_chunks = self.embedder.augmentation_metadonne(all_chunks)
        self.chro_db.save(all_chunks)

        print(f"[INFO] Base vectorielle créée et sauvegardée dans {self.chro_db.directory}")
        

    def lancement_RAG(self,llm_model: str, llm_retriever_model: str):
        self.rag = rg.RAG(self.device,self.embed_model, llm_model, llm_retriever_model)
        embedding_data = self.chro_db.load()
        self.rag.build_data_rag(embedding_data)
        self.rag.build_pipeline_rag()
        self.rag.build_retriever()

    
    def question_reponse_rag(self, query: str):
        response = self.rag.chat_rag(query)
        return response
        
    def print_all_chuncks(self):
        self.chro_db.write_all_chunks()




if __name__ == "__main__":
    app = module_app()
    app.telechargement(gf.chemindossier())
    app.lancement_RAG(app.device,app.chro_db, app.embedder, llm = mode_oll.model_Ollama(0), llm_retriever = mode_oll.model_Ollama(0), prompt_model = prompt.Prompt(1) )
    print("[INFO] Module_app exécuté avec succès.")