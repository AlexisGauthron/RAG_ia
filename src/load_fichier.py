# utils.py
from pathlib import Path
import os
import glob
from typing import List, Tuple
import uuid

# Tout les loadeurs de documents
from langchain.document_loaders import (
    TextLoader,            # Pour .txt
    UnstructuredMarkdownLoader,  # Pour .md
    PythonLoader,          # Pour .py
    JSONLoader,            # Pour .json
    PyPDFLoader,           # Pour .pdf
    UnstructuredWordDocumentLoader,  # Pour .docx
    UnstructuredExcelLoader,         # Pour .xlsx
    UnstructuredPowerPointLoader,    # Pour .pptx
    UnstructuredHTMLLoader,          # Pour .html/.htm
    UnstructuredEmailLoader,         # Pour .eml
    CSVLoader,             # Pour .csv
    UnstructuredFileLoader # Pour fichiers divers (générique)
)

# Types de fichiers supportés
LISTE_FICHIER_ACCEPTE = [".txt", ".md", ".py", ".json", ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm", ".eml", ".csv"]

LISTE_ACTUEL = [".txt", ".md", ".py", ".json", ".pdf"]  # Extensions actuellement utilisées
EXTENSION_LOADER_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": PythonLoader,
    ".json": JSONLoader,
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".eml": UnstructuredEmailLoader,
    ".csv": CSVLoader,
}


def chemindossier() -> str:
    base = Path(__file__).resolve().parent # adapte si besoin
    sous_dossier = base / "data"     # Descend dans src/data/pdf/
    return str(sous_dossier)


CHEMIN_FICHIER = chemindossier()


# Récupère tous les fichiers (récursivement)
def find_all_path_files(data_dir):
    return glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)


def find_all_files(data_dir):
    files = []
    for path_file in find_all_path_files(data_dir):
        files.append(os.path.basename(path_file))
    return files




# Charge tous les fichiers .txt/.md/.py d'un dossier.
def load_text_files(data_dir: str = f"{CHEMIN_FICHIER}/data_rag") -> List[Tuple[str, str]]:
    """
    Charge tous les fichiers .txt/.md/.py d'un dossier.
    Retourne une liste [(path, text), ...]
    """
    
    # Récupère tous les fichiers (récursivement)
    all_files = find_all_path_files(data_dir)

    # Filtre selon l'extension
    paths = []
    if not all_files:
        print(f"[WARN] Aucun fichier trouvé dans le répertoire {data_dir}. \nVeuillez vérifier le chemin : {os.path.abspath(data_dir)}")
    else:
        for f in all_files:
            if os.path.splitext(f)[1] in LISTE_FICHIER_ACCEPTE:
                paths.append(f)
                print("[INFO] Fichier trouvé:", f)
            else:
                print("[WARN] Fichier ignoré (extension non supportée):", f)

    docs = []
    
    for p in paths:
        doc = []
        # Choix du loader selon l'extension
        extension = os.path.splitext(p)[1]
        loader_cls = EXTENSION_LOADER_MAP.get(extension)
        if loader_cls:
            loader = loader_cls(p)
            
        try:
            doc = loader.load()
        except Exception as e:
            print(f"[WARN] Impossible de lire {p}: {e}")
        # print(doc,"\n\n")
        docs.append(doc)

    return docs



def save_uploaded_file(f, subdir: str = "default", dossier = chemindossier()) -> Path:
    """Enregistre un UploadedFile dans UPLOAD_ROOT/subdir avec un nom unique."""
    target_dir = f"{dossier}/{subdir}"
    os.makedirs(target_dir, exist_ok=True) 

    # Nom de fichier sécurisé + suffix conservé
    stem = Path(f.name).stem[:80]
    suffix = Path(f.name).suffix.lower()
    unique = uuid.uuid4().hex[:8]
    filename = f"{stem}-{unique}{suffix}"

    path = f"{target_dir}/{filename}"
    f.seek(0)  # s’assure qu’on écrit depuis le début
    with open(path, "wb") as out:
        out.write(f.read())
    return path

