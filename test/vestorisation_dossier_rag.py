import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import src.front.cli as cli


if __name__ == "__main__":
    Interface = cli.CLI()
    try:
        Interface.chargement_dossier_chromadb()
    except Exception as e:
        print(e)         # affiche seulement: "ValueError: …" ou juste le message selon str(e)
        sys.exit(1)      # code d'erreur, aucune traceback

