import os
from langchain_ollama.llms import OllamaLLM

# Optionnel: définir la variable d'environnement pour le modèle Ollama
# os.environ["OLLAMA_MODEL"] = "mistral"  # ou "mistral-7b-instruct" selon le nom exact
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # endpoint local d'Ollama

class model_Ollama():

    def __init__(self,index_model):
        model_ollama = ["llama3.2:3b","llama3.2:1b"]
        # Chargement du modèle Ollama via LangChain
        self.ollama_model = OllamaLLM(model=model_ollama[index_model], base_url=os.getenv("OLLAMA_HOST"))
        pass
    
    def generate(self,query):
        # Exemple d'utilisation pour générer une réponse
        response = self.ollama_model.invoke(query)

        print("Réponse du modèle Ollama:")
        return response
    
    def get_pipeline(self):
        return self.ollama_model


if __name__ == "__main__":

    import utilisation_GPU as test_GPU
    device = test_GPU.test_utilisation_GPU()

    llm = model_Ollama(0)

    print("\n Tapez 'exit' pour quitter.")
    while True:
        prompt = input("\nVous: ")
        if prompt.lower() == "exit":
            break
        response = llm.generate(prompt)
        print("\nLLM:", response)
