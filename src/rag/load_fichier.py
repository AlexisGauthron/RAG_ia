import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pathlib import Path
from typing import List, Tuple

# Document loaders - utiliser langchain_community (nouvelle API)
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PythonLoader,
    JSONLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredEmailLoader,
    CSVLoader,
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

import src.gestionnaire_fichier as gf
from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()







# Charge tous les fichiers .txt/.md/.py d'un dossier.
def load_text_files(data_dir: str = f"{CHEMIN_FICHIER}/data_rag") -> List[Tuple[str, str]]:
    """
    Charge tous les fichiers .txt/.md/.py d'un dossier.
    Retourne une liste [(path, text), ...]
    """
    
    # Récupère tous les fichiers (récursivement)
    all_files = gf.find_all_path_files(data_dir)

    # Filtre selon l'extension
    paths = []
    if not all_files:
        erreur = ValueError(f"[WARN] Aucun fichier trouvé dans le répertoire {data_dir}. \nVeuillez vérifier le chemin : {os.path.abspath(data_dir)}")
        raise erreur
    else:
        for f in all_files:
            if os.path.splitext(f)[1] in LISTE_FICHIER_ACCEPTE:
                paths.append(f)
                print("[INFO] Fichier trouvé:", f)
            else:
                print("[WARN] Fichier ignoré (extension non supportée):", f)

    docs = []

    for p in paths:
        extension = os.path.splitext(p)[1].lower()
        loader_cls = EXTENSION_LOADER_MAP.get(extension)

        if not loader_cls:
            print(f"[WARN] Pas de loader pour l'extension '{extension}': {p}")
            continue

        try:
            loader = loader_cls(p)
            doc = loader.load()
            if doc:
                docs.append(doc)
        except Exception as e:
            print(f"[WARN] Impossible de lire {p}: {e}")

    return docs



# Taille maximale autorisée (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


def save_uploaded_file(f, subdir: str = "default", dossier=chemindossier()) -> Path:
    """
    Enregistre un UploadedFile de manière sécurisée.

    Validations effectuées:
    - Extension du fichier autorisée
    - Taille du fichier <= MAX_FILE_SIZE
    - Nom de fichier sanitisé

    Args:
        f: Fichier uploadé (Streamlit UploadedFile)
        subdir: Sous-dossier de destination
        dossier: Dossier racine

    Returns:
        Path du fichier sauvegardé

    Raises:
        ValueError: Si le fichier ne passe pas les validations
    """
    target_dir = f"{dossier}/{subdir}"
    os.makedirs(target_dir, exist_ok=True)

    # Validation de l'extension
    suffix = Path(f.name).suffix.lower()
    if suffix not in LISTE_ACTUEL:
        raise ValueError(f"Extension '{suffix}' non autorisée. Extensions valides: {LISTE_ACTUEL}")

    # Validation de la taille
    f.seek(0, 2)  # Aller à la fin du fichier
    file_size = f.tell()
    f.seek(0)  # Revenir au début

    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"Fichier trop volumineux ({file_size / 1024 / 1024:.1f} MB). Maximum: {MAX_FILE_SIZE / 1024 / 1024} MB")

    if file_size == 0:
        raise ValueError("Le fichier est vide")

    # Sanitisation du nom de fichier (supprime les caractères dangereux)
    import re
    stem = Path(f.name).stem[:80]
    stem = re.sub(r'[^\w\-]', '_', stem)  # Garde uniquement lettres, chiffres, underscore, tiret
    filename = f"{stem}{suffix}"

    path = f"{target_dir}/{filename}"
    with open(path, "wb") as out:
        out.write(f.read())

    print(f"[INFO] Fichier sauvegardé: {filename} ({file_size / 1024:.1f} KB)")
    return path


