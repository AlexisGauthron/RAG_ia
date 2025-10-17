import os
from langchain_ollama.llms import OllamaLLM

# Optionnel: définir la variable d'environnement pour le modèle Ollama
# os.environ["OLLAMA_MODEL"] = "mistral"  # ou "mistral-7b-instruct" selon le nom exact
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # endpoint local d'Ollama

from typing import Optional, Iterable, Union, Literal

class model_Ollama():

    def __init__(self,model_ollama,
                    format: Literal['', 'json'] = '',   # '' = texte libre (pas de contrainte JSON)
                    temperature: float = 0.8,
                    top_p: float = 0.9,
                    top_k: int = 40,
                    mirostat: int = 0,                  # 0 = OFF
                    mirostat_eta: float = 0.1,
                    mirostat_tau: float = 5.0,
                    repeat_penalty: float = 1.1,
                    repeat_last_n: int = 64,
                    num_predict: int = -1,              # -1 = pas de limite explicite
                    seed: Optional[int] = 0,            # 0 = non fixé / non déterministe
                    stop: Optional[Iterable[str]] = None
                ):
        
        # Chargement du modèle Ollama via LangChain
        self.ollama_model = OllamaLLM(model=model_ollama, 
                                      base_url=os.getenv("OLLAMA_HOST"), 
                                      format = format,
                                      temperature=temperature,
                                      top_p=top_p,
                                      top_k=top_k,
                                      mirostat=mirostat,
                                      mirostat_eta=mirostat_eta,
                                      mirostat_tau=mirostat_tau,
                                      repeat_penalty=repeat_penalty,
                                      repeat_last_n=repeat_last_n,
                                      num_predict=num_predict,
                                      seed=seed,
                                      stop=stop
                                    )
        

        print("[INFO] Chargement model Ollama")
        pass
    
    def generate(self,query):
        # Exemple d'utilisation pour générer une réponse
        response = self.ollama_model.invoke(query)

        print("Réponse du modèle Ollama:")
        return response
    
    def get_pipeline(self):
        return self.ollama_model

