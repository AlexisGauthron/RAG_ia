from pathlib import Path
import shutil
import glob
import os


def chemindossier() -> str:
    base = Path(__file__).resolve().parent.parent # adapte si besoin
    sous_dossier = base / "data"     # Descend dans src/data/pdf/
    return str(sous_dossier)


def switch_directory(chemin_entree, chemin_sortie):
    
    entree = Path(chemin_entree)
    sortie = Path(chemin_sortie)
    print(f"[DEBUG] Destination PATH parametre: {sortie}")
    sortie.mkdir(parents=True, exist_ok=True)

    for file_path in find_all_path_files(entree):
        print(f"\n[DEBUG] Path file :{file_path}\n")
        source_path = Path(file_path)
        destination_path = sortie / source_path.name
        print(f"[DEBUG] Destination PATH : {destination_path}")
        shutil.copy2(source_path, destination_path)
        
    shutil.rmtree(entree, ignore_errors=False)


# Récupère tous les fichiers (récursivement)
def find_all_path_files(data_dir):
    return glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)


def find_all_files(data_dir):
    files = []
    for path_file in find_all_path_files(data_dir):
        files.append(os.path.basename(path_file))

    print(f"[INFO] Fichier trouvé dans le dossier {data_dir} :\n {files}\n")
    return files
