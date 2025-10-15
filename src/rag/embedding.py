import sys
import os
from datetime import datetime

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Importations standard
import re
import json
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from datetime import datetime
import os
from typing import List, Dict

# Importations internes
import src.rag.load_fichier as lf
from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()



def chunk_text(text: str, chunk_size: int = 800 , chunk_overlap: int = 120 ) -> List[str]:
    """
    Découpe un texte en morceaux qui se recouvrent légèrement.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,         # longueur max de chaque segment (en caractères)
        chunk_overlap=chunk_overlap,       # chevauchement entre les segments
        separators=["\n\n", "\n", ".", " "]  # ordre de priorité pour les coupures
    )
    chunks = splitter.split_documents(text)
    # print("\n\n",chunks,"\n\n")

    return chunks


def augmentation_metadonne(chunks: List[Dict]) -> List[Dict]:
    """
    Ajoute des métadonnées supplémentaires à chaque chunk dans la liste.
    Par exemple : date de traitement, nom et extension du fichier.

    Args:
        chunks (List[Dict]): Liste de chunks avec un champ 'metadata' dict.
        path_file (str): Chemin complet du fichier source pour extraire nom et extension.

    Returns:
        List[Dict]: Liste mise à jour avec métadonnées augmentées.
    """
    
    metadonne_augmentee = []
    timestamp = datetime.utcnow().isoformat() + "Z"  # horodatage ISO UTC

    # print("CHUNKS",chunks)


    for chunk in chunks:
        # Récupérer les métadonnées existantes ou initialiser un dict vide
        metadata = chunk['metadata']
    
        nom_fichier = metadata["source"]
        extension_fichier = os.path.splitext(nom_fichier)[1].replace('.', '').upper()  # extension sans point, en MAJ
        nom_fichier = os.path.basename(nom_fichier)

        # Ajouter ou mettre à jour les métadonnées spécifiques
        metadata.update({
            'date_ajout': timestamp,
            # 'nom_fichier': nom_fichier,
            'extension_fichier': extension_fichier,
            'source': nom_fichier
        })

        # Mettre à jour le chunk avec ces métadonnées augmentées
        chunk['metadata'] = metadata

        metadonne_augmentee.append(chunk)

    return metadonne_augmentee





# Embedding avec HuggingFace 
class Embedding_datasource:
    def __init__(self):
        pass


    def build_chunk(self, doc: Tuple[str, str], chunk_size: int = 800 , chunk_overlap: int = 120):
        # Récupère le texte et les métadonnées
        chunks = chunk_text(doc, chunk_size, chunk_overlap)

        # Sauvegarde exemple de chunks pour pouvoir créer automatiquement les paramètres 
        self.save_ex_chunks(chunks)

        return chunks


    def build_all_chunks(self, docs: List[Tuple[str, str]]):
        # Découpe chaque document en chunks
        all_chunks = []
        for doc in docs:
            chunks = self.build_chunk(doc)
            all_chunks.extend(chunks)
            
        print(f"[INFO] {len(all_chunks)} chunks créés à partir de {len(docs)} documents.")
        
        return all_chunks
    










     # BONUSSSSSS : Sauvegarde d'exemples de chunks
    



    def save_ex_chunks(self, chunks):
        """
        Sauvegarde uniquement les 2 premiers chunks dans un fichier JSON,
        nommé d'après le champ metadata['source'] du premier chunk.
        Crée automatiquement le dossier chroma_res/ si nécessaire.
        """
        try:
            # ✅ Vérification de la présence de chunks
            if not chunks:
                print("[WARN] Aucun chunk à sauvegarder.")
                return

            # ✅ Création du dossier s'il n'existe pas
            path_dos = os.path.dirname(f"{CHEMIN_FICHIER}/chroma_res/")
            os.makedirs(path_dos, exist_ok=True)

            # ✅ Récupération de la source depuis le metadata du premier chunk
            source_path = chunks[0].metadata.get("source", "unknown_source")

            # Extraction et nettoyage du nom du fichier
            base_name = os.path.basename(source_path)
            file_name = os.path.splitext(base_name)[0]
            file_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', file_name)

            # ✅ Définir le chemin final du fichier JSON
            filename = f"{path_dos}/{file_name}_chunks.json"

            # ✅ Limiter à 2 chunks
            chunks_to_save = chunks[:2]

            # ✅ Sauvegarde des chunks dans le fichier
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            # "page_content": c.page_content,
                            "metadata": c.metadata
                        } for c in chunks_to_save
                    ],
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            print(f"[INFO] {len(chunks_to_save)} chunks sauvegardés dans {filename}")

        except Exception as e:
            print(f"[ERROR] Erreur lors de la sauvegarde des chunks: {e}")



    def list_metadata(self, docs):
        """Retourne toutes les clés de métadonnées distinctes présentes dans docs."""

        def walk(item):
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    walk(sub)
                return
            if hasattr(item, "metadata"):
                md = getattr(item, "metadata", {}) or {}
                if isinstance(md, dict):
                    self.metadata.update(md.keys())

        walk(docs)


    def get_metadata(self):
        return sorted(self.metadata)
    



####### Lancement de tests #######

import test.utilisation_GPU as test_GPU


if __name__ == "__main__":
    # Test et utilisation du GPU si disponible
    device = test_GPU.test_utilisation_GPU()

    Embedding = Embedding_datasource(device)
    # Embedding.run()
    print("\n ####### METADATA #######\n",Embedding.get_metadata())
    # print(Embedding.get_metadata())


