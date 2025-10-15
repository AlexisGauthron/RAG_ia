import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import src.front.cli as cli

Interface = cli.CLI()
# Interface.chargement_dossier_chromadb()
# Interface.write_chunk()


# test_suppression = ["Horama INGE 4.pdf","GUIDE_ULTIME_INVESTISSEMENT-IMMOBILIER_DUR.pdf"]
# for file in test_suppression:
#     Interface.delete_files(file)

# Interface.write_chunk()

Interface.delete_all_files()
# Interface.write_chunk()