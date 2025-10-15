from pathlib import Path
import os

# Fichier inclue dans module_app.py
import test.utilisation_GPU as test_GPU
import src.rag.load_fichier as lf
import src.rag.embedding as emb
import src.rag.chroma_database as chdt
import src.rag.rag as rg

import src.modele.modele_Embeddings as modele_Emb
import src.modele.modele_LLM_ollama as mode_oll

import src.rag.prompt as prompt

import src.gestionnaire_fichier as gf

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()
CHEMIN_FICHIER_RAG = f"{CHEMIN_FICHIER}/data_rag"


class module_app:
    def __init__(self, embed_model, prompt_model: int, directory: str = CHEMIN_FICHIER_RAG):
        self.device = test_GPU.test_utilisation_GPU()
        self.embed_model = modele_Emb.Model_embeddings(self.device,embed_model).get_embedder()
        self.prompt_model = prompt_model
        self.directory_data_rag = directory
        self.directory_importer = f"{CHEMIN_FICHIER}/Importer"
         # Initialisation des composants RAG

        self.embedder = emb.Embedding_datasource()
        self.chromadb = chdt.ChromaDB(self.embed_model)
        self.rag = None



    # Fonction pour créer les embeddings et la base vectorielle
    def telechargement(self):
        data_folder = self.directory_importer
        data_folder_path = Path(data_folder)
        data_folder_path.mkdir(parents=True, exist_ok=True)

        for file in gf.find_all_path_files(data_folder):
            print(f"[DEBUG] Nom fichier :{file}\n")
            doublons = self.chromadb.delete_files(os.path.basename(file),check_doublons=True)

        docs = lf.load_text_files(data_folder)
        all_chunks = self.embedder.build_all_chunks(docs)
        
        all_chunks = chdt.documents_to_dict(all_chunks)

        # Augmentation des métas données
        all_chunks = emb.augmentation_metadonne(all_chunks)
        self.chromadb.save(all_chunks)
        print(f"\n[DEBUG] Parametre chemin : {self.directory}\n")
        gf.switch_directory(data_folder,self.directory)
        print(f"[INFO] Base vectorielle créée et sauvegardée dans {self.chromadb.directory}")
        self.chromadb.write_all_chunks()
        print(f"[INFO] Chunks ecrit dans data/all_chunks/all_chunks.json")
        return doublons




    def delete_files(self,nom_fichier):
        delete = self.chromadb.delete_files(nom_fichier)
        if delete == 1:
            print("Suprresion fichier !!!\n")
            chemin_complet = Path(f"{CHEMIN_FICHIER_RAG}/{nom_fichier}")
            if chemin_complet.is_file():
                chemin_complet.unlink()
        self.chromadb.write_all_chunks()
        print(f"[INFO] Chunks ecrit dans data/all_chunks/all_chunks.json")


    def lancement_RAG(self,llm_model: str, llm_retriever_model: str):
        self.rag = rg.RAG(self.device,self.embed_model, llm_model, llm_retriever_model)
        embedding_data = self.chromadb.load()
        self.rag.build_data_rag(embedding_data)
        self.rag.build_pipeline_rag()
        self.rag.build_retriever()

    
    def question_reponse_rag(self, query: str):
        response = self.rag.chat_rag(query)
        return response
        
    




if __name__ == "__main__":
    app = module_app()
    app.telechargement(gf.chemindossier())
    app.lancement_RAG(app.device,app.chromadb, app.embedder, llm = mode_oll.model_Ollama(0), llm_retriever = mode_oll.model_Ollama(0), prompt_model = prompt.Prompt(1) )
    print("[INFO] Module_app exécuté avec succès.")