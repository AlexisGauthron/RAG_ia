from pathlib import Path

def chemindossier() -> str:
    base = Path(__file__).resolve().parent.parent # adapte si besoin
    sous_dossier = base / "data"     # Descend dans src/data/pdf/
    return str(sous_dossier)

