import embedding as emb
import modele_LLM as mod
import utilisation_GPU as test_GPU
import prompt as prompt
import modele_Embeddings as modele_Emb
import vectoriel_research as vec


from path_file import chemindossier
CHEMIN_FICHIER = chemindossier()


from typing import Dict

from transformers import pipeline
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None
try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class RAG:
    def __init__(self):
        # Test et utilisation du GPU si disponible
        device = test_GPU.test_utilisation_GPU()

        # Initialisation de la création des embeddings et de la base vectorielle
        self.embedder = modele_Emb.modele_embedding[0]  # Choisir le modèle d'embedding (0 ou 1)
        print(f"[INFO] Modèle d'embedding sélectionné : {self.embedder['model']}")
        self.embed_data = emb.Embedding_datasource(device,self.embedder["model"])
        self.embedding_data = None

        # Initialisation des modèle de langage
        self.llm = mod.QwenLLM(device=device, quantized=False)
        self.llm_retriever = mod.Mistral7BLLM(device=device, quantized=False)

        
        # Modèle de prompt simple
        self.prompt = prompt.Prompt(1)
        self.vestor_research = None
        self.retriever = None
        self.rag = None


    # Création des embeddings et de la base vectorielle
    def build_data_rag(self):
        self.embedding_data = self.embed_data.run(chunk_max_size=800, chunk_overlap=120)
        

    def build_retriever(self):
        if not self.embedding_data:
            print("[WARN] L'index n'est pas construit.")
            return []
        self.vestor_research = vec.Vectoriel_research(self.embedding_data)

        # self.vestor_research.search(top_k=k)
        self.vestor_research.search_llm(self.llm_retriever.get_pipeline())
        self.retriever = self.vestor_research.get_retriever()

        # self.retriever = self.embedding_data.as_retriever(search_kwargs={"k": k})
    

    # 4️⃣ Création du pipeline RAG
    def build_pipeline_rag(self):
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

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


    # 5️⃣ Boucle d'interaction
    def chat_with_rag(self):
        print("Posez vos questions (tapez 'exit' pour quitter) :")

        while True:
            try:
                question = input("Vous: ")
                if question.lower() == "exit":
                    break
                
                result = self.rag.invoke({"query": question})

                print("📘 Question :", question)
                print("💬 Réponse :", result["result"])
                print("Sources utilisées :")
                for doc in result["source_documents"]:
                    print(f"Source: {doc.metadata.get('source', 'inconnu')} — extrait: {doc.page_content[:100]}")


                    
            except KeyboardInterrupt:
                print("\n(Interruption) Tapez ':exit' pour quitter.")
            except Exception as e:
                print(f"[ERR] {type(e).__name__}: {e}")




if __name__ == "__main__":
    rag = RAG()
    rag.build_data_rag()
    rag.build_pipeline_rag()
    rag.chat_with_rag()

