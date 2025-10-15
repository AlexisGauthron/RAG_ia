import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from pathlib import Path

from src.modele import modele_LLM_hugface as mod_hug

import src.modele.modele_LLM_ollama as mode_oll
import src.modele.modele_Embeddings as modele_Emb

import src.rag.embedding as emb
import src.rag.prompt as prompt
import src.rag.vectoriel_research as vec
import test.utilisation_GPU as test_GPU
import src.rag.chroma_database as chdt

import src.front.module_app as mapp

import src.gestionnaire_fichier as gf
import src.rag.load_fichier as lf

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()
CHEMIN_FICHIER_RAG = f"{chemindossier()}/data_rag"
CHEMIN_FICHIER_DATABASE = f"{chemindossier()}/chroma_db"


class CLI:

    def __init__(self):
        self.device = test_GPU.test_utilisation_GPU()
        self.directory_file_rag = CHEMIN_FICHIER_RAG 
        self.directory_data_rag = CHEMIN_FICHIER_DATABASE
        self.model_embedder = modele_Emb.Model_embeddings(self.device,0)
        self.f_embeding = emb.Embedding_datasource()
        self.chromadb = chdt.ChromaDB(self.model_embedder.get_embedder(),self.directory_data_rag)
        # self.embeding = None
        pass


    def chargement_dossier_chromadb(self):
        print("[INFO] Initialisation Chargement fichier\n")
        data_folder = self.directory_file_rag
        data_folder_path = Path(data_folder)
        data_folder_path.mkdir(parents=True, exist_ok=True)

        for file in gf.find_all_files(data_folder):
            self.chromadb.delete_files(file)

        # Chargement des données et sauvegarde dans la database
        docs = lf.load_text_files(data_folder)
        all_chunks = self.f_embeding.build_all_chunks(docs)
        # Transformer la liste d'objets Document en liste de dict {text, metadata}
        all_chunks = [{'text': doc.page_content, 'metadata': doc.metadata} for doc in all_chunks]
        # Augmentation des métas données
        all_chunks = emb.augmentation_metadonne(all_chunks)
        self.chromadb.save(all_chunks)

        # Si les deux dossiers différents 
        if not os.path.samefile(data_folder, self.directory_file_rag):
            gf.switch_directory(data_folder,self.directory_file_rag)

        print(f"[INFO] Base vectorielle créée et sauvegardée dans {self.directory_data_rag}")



        
    def delete_files(self,nom_fichier):
        print(f"[INFO] Suppression fichier {nom_fichier}\n")
        delete = self.chromadb.delete_files(nom_fichier)
        if delete == 1:
            print("Suprresion fichier !!!\n")
            chemin_complet = Path(f"{CHEMIN_FICHIER_RAG}/{nom_fichier}")
            if chemin_complet.is_file():
                chemin_complet.unlink()


    def delete_all_files(self):
        self.chromadb.delete_all()

    def lancement_RAG(self,llm_model: str, llm_retriever_model: str):
        self.rag = rg.RAG(self.device,self.embed_model, llm_model, llm_retriever_model)
        embedding_data = self.chro_db.load()
        self.rag.build_data_rag(embedding_data)
        self.rag.build_pipeline_rag()
        self.rag.build_retriever()


    def write_chunk(self):
        self.chromadb.write_all_chunks()