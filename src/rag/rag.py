import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Imports des modules internes
import src.modele.modele_LLM_ollama as modele_oll
import src.rag.prompt as Prompt
import src.rag.vectoriel_research as vec

from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()

class RAG:
    def __init__(self, device, embedder, llm, llm_retriever, prompt_model, mode, top_k=6):

        self.device = device
        self.embedder = embedder
        self.llm = modele_oll.model_Ollama(llm, temperature=0.3)
        self.llm_retriever = modele_oll.model_Ollama(llm_retriever, temperature=0.8)
        self.prompt = prompt_model
        self.mode = mode
        self.top_k = top_k

        self.embedding_data = None
        self.vector_research = None
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
            llm_retriever = modele_oll.model_Ollama(modele_llm_retriever)


        if not self.embedding_data:
            message_erreur = ValueError("[WARN] L'index n'est pas construit.")
            raise message_erreur

        self.vector_research = vec.Vectoriel_research(self.embedding_data, embedder=self.embedder)
        if mode_filtre == "default":
            if self.mode == "default":
                self.vector_research.search(top_k=self.top_k, llm=self.llm.get_pipeline())
            else:
                self.vector_research.search_llm(llm_retriever.get_pipeline())
        else:
            self.vector_research.search_llm(llm_retriever.get_pipeline())

        self.retriever = self.vector_research.get_retriever()

    def build_pipeline_rag(self, index_prompt=-1, modele_llm="default", mode_filtre="default"):
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        # Choix Prompt
        if index_prompt == -1:
            prompt = self.prompt.get_prompt()
        else: 
            prompt = Prompt.Prompt(index_prompt).get_prompt()

        # print(f"[INFO] Prompt utilisé : {prompt.template}")

        if modele_llm == "default":
            llm = self.llm
        else:
            llm = modele_oll.model_Ollama(modele_llm)
        
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
    def chat_with_rag_console(self, selection_chunk="default", mode_filtre="default"):
        """Boucle d'interaction console pour le RAG."""
        print("Posez vos questions (tapez 'exit' pour quitter) :")

        while True:
            try:
                question = input("Vous: ")
                if question.lower() == "exit":
                    break

                result = self.chat_rag(question, mode_filtre)

                print("Question :", question, "\n")
                print("Reponse :", result["result"], "\n")

                print("Sources utilisees :")
                if selection_chunk != "default":
                    self.chunks_selectionne_with_score(question)
                else:
                    self.chunks_selectionne_unique(result)
                print("\n")

            except KeyboardInterrupt:
                print("\n(Interruption) Tapez 'exit' pour quitter.")
            except Exception as e:
                print(f"[ERR] {type(e).__name__}: {e}")
