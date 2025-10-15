import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import src.rag.chroma_database as db
import src.modele.modele_Embeddings as modele_Emb
import test.utilisation_GPU as test_GPU

device = test_GPU.test_utilisation_GPU()
embed_model = 0
modele_emb = modele_Emb.Model_embeddings(device,embed_model).get_embedder()
databa = db.ChromaDB(modele_emb)

databa.mise_a_jour_metadata()

# print(databa.get_chunks_db())

databa.write_all_chunks()

