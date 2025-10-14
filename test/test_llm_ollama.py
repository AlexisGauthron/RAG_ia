import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import test.utilisation_GPU as test_GPU
from src.modele import modele_LLM_ollama as mod_oll



if __name__ == "__main__":

    
    device = test_GPU.test_utilisation_GPU()

    llm = mod_oll.model_Ollama(0)

    print("\n Tapez 'exit' pour quitter.")
    while True:
        prompt = input("\nVous: ")
        if prompt.lower() == "exit":
            break
        response = llm.generate(prompt)
        print("\nLLM:", response)