import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import src.front.cli as cli

Interface = cli.CLI()

model_ollama = ["llama3.2:3b","llama3.2:1b","mistral:7b-instruct","deepseek-r1:8b"]

selection_chunk = ["default","score"]


# Interface.Rag(index_prompt=-1,modele_llm="mistral:7b-instruct")


Interface.Rag(index_prompt=-1,modele_llm="mistral:7b-instruct",modele_llm_retriever="mistral:7b-instruct",mode_filtre="filtre")