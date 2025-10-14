import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from src.modele import modele_LLM_hugface as mod_hug

import src.modele.modele_LLM_hugface as mod_hug
import src.modele.modele_LLM_ollama as mode_oll
import src.modele.modele_Embeddings as modele_Emb

import src.rag.embedding as emb
import test.utilisation_GPU as test_GPU
import src.rag.prompt as prompt
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



class RAG:
    def __init__(self, device, embedder = modele_Emb.modele_embedding[0], llm = mode_oll.model_Ollama(0), llm_retriever = mode_oll.model_Ollama(0), prompt_model = prompt.Prompt(1), mode: str = "default"):

        self.device = device  
        self.embedder = embedder  
        self.llm = llm
        self.llm_retriever = llm_retriever
        self.prompt = prompt_model
        self.mode = mode

        self.embedding_data = None
        self.vestor_research = None
        self.retriever = None
        self.rag = None
        


    # Création des embeddings et de la base vectorielle
    def build_data_rag(self,embedding_data):
        self.embedding_data = embedding_data
        

    def build_retriever(self):
        if not self.embedding_data:
            print("[WARN] L'index n'est pas construit.")
            return []
        self.vestor_research = vec.Vectoriel_research(self.embedding_data)
        # print("\n[DEBUG] model",self.llm_retriever)

        if self.mode == "default":
            self.vestor_research.search(top_k= 5)  
        else:
            self.vestor_research.search_llm(self.llm_retriever.get_pipeline())

        self.retriever = self.vestor_research.get_retriever()
    

        # self.retriever = self.embedding_data.as_retriever(search_kwargs={"k": k})
    

    # 4️⃣ Création du pipeline RAG
    def build_pipeline_rag(self):
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        print(self.prompt)
        prompt = self.prompt.get_prompt()
        print(f"[INFO] Prompt utilisé : {prompt.template}")
        self.build_retriever()

        # Créer la chaîne RAG
        self.rag = RetrievalQA.from_chain_type(
            llm=self.llm.get_pipeline(),
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )


    def chat_rag(self, query: str):
        if not self.rag:
            print("[WARN] Le pipeline RAG n'est pas construit.")
            return None
        
        if self.mode != "default":
            self.check_rag_filtre(query)  # Optionnel : pour déboguer les filtres

        return self.rag.invoke({"query": query})


    def check_rag_filtre(self, query: str):
        # 🧩 --- Ici on capture le filtre avant qu’il soit utilisé ---
        try : 

            structured_query_obj = self.retriever.query_constructor.invoke(query)
            print("\n[DEBUG] Structure Query", structured_query_obj)
            print("\n[DEBUG] Filtre structuré généré : ", structured_query_obj.filter)
            print("\n[DEBUG] Requête interprétée :", structured_query_obj.query)
        except Exception as e:
            print(f"[ERR_CHECK_RAG_FILTRE] {type(e).__name__}: {e}")
            return None


    # 5️⃣ Boucle d'interaction
    def chat_with_rag_console(self):
        print("Posez vos questions (tapez 'exit' pour quitter) :")

        while True:
            # try:
                question = input("Vous: ")
                if question.lower() == "exit":
                    break
                
                if self.mode != "default":
                    self.check_rag_filtre(question)  # Optionnel : pour déboguer les filtres

    
                result = self.rag.invoke({"query": question})

                print("📘 Question :", question,"\n\n")
                print("💬 Réponse :", result["result"],"\n\n")

                # Affichage des sources (uniques)
                print("📚 Sources utilisées :")
                seen_sources = set()
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "inconnu")
                    if source not in seen_sources:
                        seen_sources.add(source)
                        excerpt = doc.page_content[:100].replace("\n", " ")
                        print(f"Source: {source} — extrait: {excerpt}")
                print("\n")


                    
            # except KeyboardInterrupt:
            #     print("\n(Interruption) Tapez ':exit' pour quitter.")
            # except Exception as e:
            #     print(f"[ERR] {type(e).__name__}: {e}")



if __name__ == "__main__":
    device = test_GPU.test_utilisation_GPU()
    rag = RAG(device)
    embed_model = modele_Emb.Model_embeddings(device,0).get_embedder()
    chro_db = chdt.ChromaDB(embed_model)
    embedding_data = chro_db.load()
    print(embedding_data)
    rag.build_data_rag(embedding_data)
    rag.build_pipeline_rag()
    rag.chat_with_rag_console()


