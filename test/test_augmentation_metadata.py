import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import src.rag.chroma_database as db
import src.modele.modele_Embeddings as modele_Emb
import test.utilisation_GPU as test_GPU
import src.rag.embedding as emb

device = test_GPU.test_utilisation_GPU()
embed_model = 0
modele_emb = modele_Emb.Model_embeddings(device,embed_model).get_embedder()
databa = db.ChromaDB(modele_emb)
embe = emb.Embedding_datasource(modele_emb)

def test_augmentation():
    all_chunk = databa.get_chunks_db()
    # print(all_chunk)
    return embe.augmentation_metadonne(all_chunk)


test_aug = test_augmentation()
# print(test_aug)

def liste_augmentation():
    return databa.all_metadata()

liste_augmentation()
