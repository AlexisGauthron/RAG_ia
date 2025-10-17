import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.modele import modele_LLM_hugface as mod_hug

import src.modele.modele_LLM_ollama as mode_oll
import src.modele.modele_Embeddings as modele_Emb

import src.rag.embedding as emb
import src.rag.prompt as Prompt
import src.rag.vectoriel_research as vec
import test.utilisation_GPU as test_GPU
import src.rag.chroma_database as chdt

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()


# from typing import Dict

# from transformers import pipeline
# try:
#     from transformers import BitsAndBytesConfig
# except ImportError:  # pragma: no cover - optional dependency
#     BitsAndBytesConfig = None
# try:  # pragma: no cover - optional dependency
#     import torch
# except ImportError:  # pragma: no cover
#     torch = None
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import src.modele.modele_LLM_ollama as modele_oll

class RAG:
    def __init__(self, device, embedder, llm, llm_retriever, prompt_model, mode):

        self.device = device  
        self.embedder = embedder  
        self.llm = modele_oll.model_Ollama(llm)
        self.llm_retriever = modele_oll.model_Ollama(llm_retriever)
        self.prompt = prompt_model
        self.mode = mode

        self.embedding_data = None
        self.vestor_research = None
        self.retriever = None
        self.rag = None
        

    def switch_mode(self,mode):
        self.mode = mode


    # Création des embeddings et de la base vectorielle
    def build_data_rag(self,embedding_data):
        self.embedding_data = embedding_data
        

    def build_retriever(self,modele_llm_retriever = "default",mode_filtre = "default"):

        if modele_llm_retriever == "default":
            llm_retriever = self.llm_retriever
        else:
            llm_retriever = mode_oll.model_Ollama(modele_llm_retriever)


        if not self.embedding_data:
            print("[WARN] L'index n'est pas construit.")
            return []
        
        self.vestor_research = vec.Vectoriel_research(self.embedding_data)
        # print("\n[DEBUG] model",self.llm_retriever)
        if mode_filtre == "default":
            if self.mode == "default":
                self.vestor_research.search(top_k= 5)  
            else:
                self.vestor_research.search_llm(llm_retriever.get_pipeline())
        else:
            self.vestor_research.search_llm(llm_retriever.get_pipeline())

        self.retriever = self.vestor_research.get_retriever()
    

        # self.retriever = self.embedding_data.as_retriever(search_kwargs={"k": k})
    

    # 4️⃣ Création du pipeline RAG
    def build_pipeline_rag(self,index_prompt = -1, modele_llm = "default", mode_filtre = "default"):
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        # Choix Prompt
        if index_prompt == -1:
            prompt = self.prompt.get_prompt()
        else: 
            prompt = Prompt.Prompt(index_prompt).get_prompt()

        print(f"[INFO] Prompt utilisé : {prompt.template}")

        if modele_llm == "default":
            llm = self.llm
        else:
            llm = mode_oll.model_Ollama(modele_llm)
        
        self.build_retriever(mode_filtre=mode_filtre)

        # Créer la chaîne RAG
        self.rag = RetrievalQA.from_chain_type(
            llm=llm.get_pipeline(),
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )


    def chat_rag(self, query: str, mode_filtre = "default"):
        if not self.rag:
            print("[WARN] Le pipeline RAG n'est pas construit.")
            return None
        if mode_filtre == "default":
            if self.mode != "default":
                self.check_rag_filtre(query)  # Optionnel : pour déboguer les filtres
        else:
            self.check_rag_filtre(query)  # Optionnel : pour déboguer les filtres

            
        return self.rag.invoke({"query": query})


    def check_rag_filtre(self, query: str):
        # Capture du filtre self-query avant usage.
        if not self.retriever:
            print("[WARN] Aucun retriever initialise, controle du filtre impossible.")
            return None

        if not hasattr(self.retriever, 'query_constructor'):
            print("[INFO] Retriever courant sans SelfQuery, saut de l'inspection du filtre.")
            return None

        try:
            structured_query_obj = self.retriever.query_constructor.invoke(query)
            print("\n[DEBUG] Structure Query", structured_query_obj)
            print("\n[DEBUG] Filtre structure genere : " , structured_query_obj.filter)
            print("\n[DEBUG] Requete interpretee :", structured_query_obj.query)
        except Exception as e:
            print(f"[ERR_CHECK_RAG_FILTRE] {type(e).__name__}: {e}")
            print(f"[ERROR] Erreur utilisation filtre\n")
            return None


    def chunks_selectionne_with_score(self,query):
        
        docs_scores = self.retriever.vectorstore.similarity_search_with_relevance_scores(query, k=5)
        for i, (doc, score) in enumerate(docs_scores, 1):
            print(f"Chunk {i} – score: {score:.3f}")
            print(f"Source: {doc.metadata.get("source")}")
            print(f"Page: {doc.metadata.get('page')}")
            print(f"Chunk:\n",doc.page_content, "\n\n")


    def chunks_selectionne_unique(self,result):
        # Affichage des sources (uniques)
            for doc in result["source_documents"]:
                print(f"Source: {doc.metadata.get("source")}")
                print(f"Page: {doc.metadata.get('page')}")
                print(f"Chunk:\n",doc.page_content, "\n\n")     
                    



    # 5️⃣ Boucle d'interaction
    def chat_with_rag_console(self,selection_chunk = "default", mode_filtre = "default"):
        print("Posez vos questions (tapez 'exit' pour quitter) :")

        while True:
            # try:
                question = input("Vous: ")
                if question.lower() == "exit":
                    break
                
                result = self.chat_rag(question,mode_filtre)

                print("📘 Question :", question,"\n\n")
                print("💬 Réponse :", result["result"],"\n\n")

            
                # Affichage des sources (uniques)
                print("📚 Sources utilisées :")
                if selection_chunk != "default":
                    self.chunks_selectionne_with_score(question)
                else:
                    self.chunks_selectionne_unique(result)
                print("\n")


                    
            # except KeyboardInterrupt:
            #     print("\n(Interruption) Tapez ':exit' pour quitter.")
            # except Exception as e:
            #     print(f"[ERR] {type(e).__name__}: {e}")



