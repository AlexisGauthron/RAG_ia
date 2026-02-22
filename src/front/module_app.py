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
    def __init__(self, embed_model, prompt_model: int, directory: str = CHEMIN_FICHIER_RAG, mode_retriever : str = "default"):
        self.device = test_GPU.test_utilisation_GPU()
        self.embed_model = modele_Emb.Model_embeddings(self.device,embed_model).get_embedder()
        self.prompt_model = prompt_model
        self.directory_data_rag = directory
        self.directory_importer = f"{CHEMIN_FICHIER}/Importer"
        self.methode_retriever = mode_retriever
         # Initialisation des composants RAG

        self.embedder = emb.Embedding_datasource()
        self.chromadb = chdt.ChromaDB(self.embed_model)
        self.rag = None

    def modifications_mode_retriever(self,mode_retriever):
        self.methode_retriever = mode_retriever

    def telechargement(self):
        """Traite les fichiers importés : supprime les doublons, crée les chunks et sauvegarde."""
        data_folder = self.directory_importer
        data_folder_path = Path(data_folder)
        data_folder_path.mkdir(parents=True, exist_ok=True)

        # Initialiser doublons à 0 (pas de doublons par défaut)
        doublons = 0
        files = gf.find_all_path_files(data_folder)

        for file in files:
            print(f"[DEBUG] Nom fichier: {file}\n")
            result = self.chromadb.delete_files(os.path.basename(file), check_doublons=True)
            # Si au moins un doublon trouvé, on le signale
            if result == 1:
                doublons = 1

        docs = lf.load_text_files(data_folder)
        all_chunks = self.embedder.build_all_chunks(docs)
        all_chunks = chdt.documents_to_dict(all_chunks)

        # Augmentation des métadonnées
        all_chunks = emb.augmentation_metadonne(all_chunks)
        self.chromadb.save(all_chunks)
        print(f"\n[DEBUG] Parametre chemin: {self.directory_data_rag}\n")
        gf.switch_directory(data_folder, self.directory_data_rag)
        print(f"[INFO] Base vectorielle créée et sauvegardée dans {self.chromadb.directory}")
        self.chromadb.write_all_chunks()
        print(f"[INFO] Chunks ecrit dans data/all_chunks/all_chunks.json")
        return doublons




    def sync_files(self):
        """Synchronise ChromaDB et le disque. Supprime les orphelins des deux cotes."""
        if self.chromadb.vectordb is None:
            self.chromadb.load()
        if self.chromadb.vectordb is None:
            return False

        changed = False

        # Verifier si les sources necessitent une normalisation (chemins complets → basenames)
        all_data = self.chromadb.vectordb._collection.get(include=["metadatas"])
        needs_normalization = any(
            m.get("source", "") != os.path.basename(m.get("source", ""))
            for m in all_data.get("metadatas", []) if m.get("source")
        )

        if needs_normalization:
            print("[SYNC] Normalisation des metadonnees source (chemins complets -> basenames)")
            self.chromadb.mise_a_jour_metadata()
            changed = True
            # Recharger les metadonnees apres normalisation
            all_data = self.chromadb.vectordb._collection.get(include=["metadatas"])

        # Sources dans ChromaDB (maintenant normalisees)
        chroma_sources = set(
            os.path.basename(m.get("source", ""))
            for m in all_data.get("metadatas", []) if m.get("source")
        )

        # Fichiers sur disque (filtrer les repertoires)
        disk_files = set(
            f for f in gf.find_all_files(self.directory_data_rag)
            if os.path.isfile(os.path.join(self.directory_data_rag, f))
        )

        # ChromaDB vide mais fichiers presents → reconstruire la base
        if not chroma_sources and disk_files:
            print("[SYNC] ChromaDB vide, reconstruction a partir des fichiers sur disque...")
            docs = lf.load_text_files(self.directory_data_rag)
            all_chunks = self.embedder.build_all_chunks(docs)
            all_chunks = chdt.documents_to_dict(all_chunks)
            all_chunks = emb.augmentation_metadonne(all_chunks)
            self.chromadb.save(all_chunks)
            self.chromadb.write_all_chunks()
            print(f"[SYNC] ChromaDB reconstruite a partir de {len(disk_files)} fichier(s)")
            return True

        # Orphelins ChromaDB → supprimer chunks
        for f in chroma_sources - disk_files:
            print(f"[SYNC] Suppression chunks orphelins: {f}")
            self.chromadb.delete_files(f)
            changed = True

        # Orphelins disque → supprimer fichier
        for f in disk_files - chroma_sources:
            print(f"[SYNC] Suppression fichier orphelin: {f}")
            chemin = Path(os.path.join(self.directory_data_rag, f))
            if chemin.is_file():
                chemin.unlink()
            changed = True

        if changed:
            self.chromadb.write_all_chunks()
            print("[SYNC] Synchronisation terminee avec corrections")
        else:
            print("[SYNC] ChromaDB et disque sont synchronises")

        return changed

    def delete_files(self, nom_fichier: str):
        """Supprime un fichier de la base vectorielle et du disque."""
        self.chromadb.delete_files(nom_fichier)
        print(f"[INFO] Suppression fichier: {nom_fichier}")

        # Toujours supprimer le fichier physique, meme s'il n'etait pas dans ChromaDB
        chemin_complet = Path(f"{CHEMIN_FICHIER_RAG}/{nom_fichier}")
        if chemin_complet.is_file():
            chemin_complet.unlink()

        self.chromadb.write_all_chunks()
        print(f"[INFO] Chunks ecrit dans data/all_chunks/all_chunks.json")


    def lancement_RAG(self, llm_model: str, llm_retriever_model: str, mode_retriever: str = None, top_k: int = 6):
        """Lance le pipeline RAG avec les modèles spécifiés."""
        if mode_retriever is not None:
            self.methode_retriever = mode_retriever

        self.rag = rg.RAG(
            self.device,
            self.embed_model,
            llm_model,
            llm_retriever_model,
            self.prompt_model,
            self.methode_retriever,
            top_k=top_k
        )
        embedding_data = self.chromadb.load()
        self.rag.build_data_rag(embedding_data)
        # build_pipeline_rag() appelle déjà build_retriever() en interne
        self.rag.build_pipeline_rag()


    
    def question_reponse_rag(self, query: str):
        response = self.rag.chat_rag(query)
        return response
        
    




if __name__ == "__main__":
    app = module_app()
    app.telechargement(gf.chemindossier())
    app.lancement_RAG(app.device,app.chromadb, app.embedder, llm = mode_oll.model_Ollama(0), llm_retriever = mode_oll.model_Ollama(0), prompt_model = prompt.Prompt(1) )
    print("[INFO] Module_app exécuté avec succès.")