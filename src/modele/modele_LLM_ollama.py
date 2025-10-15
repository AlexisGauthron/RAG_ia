import os
from langchain_ollama.llms import OllamaLLM

# Optionnel: définir la variable d'environnement pour le modèle Ollama
# os.environ["OLLAMA_MODEL"] = "mistral"  # ou "mistral-7b-instruct" selon le nom exact
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # endpoint local d'Ollama

class model_Ollama():

    def __init__(self,model_ollama):
        # Chargement du modèle Ollama via LangChain
        self.ollama_model = OllamaLLM(model=model_ollama, base_url=os.getenv("OLLAMA_HOST"))
        print("[INFO] Chargement model Ollama")
        pass
    
    def generate(self,query):
        # Exemple d'utilisation pour générer une réponse
        response = self.ollama_model.invoke(query)

        print("Réponse du modèle Ollama:")
        return response
    
    def get_pipeline(self):
        return self.ollama_model

