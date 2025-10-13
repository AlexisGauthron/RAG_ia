from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from langchain.chains.query_constructor.base import AttributeInfo
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import src.modele_LLM_hugface as mod

@dataclass
class MetadataSample:
    key: str
    values: List[str]


def load_metadata_samples(directory: Path, max_values: int = 6) -> List[MetadataSample]:
    samples: Dict[str, set] = defaultdict(set)

    for json_path in directory.glob("*.json"):
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for chunk in payload:
            meta = chunk.get("metadata", {})
            for key, value in meta.items():
                samples[key].add(str(value))
                if len(samples[key]) >= max_values:
                    continue

    return [
        MetadataSample(key=key, values=sorted(list(values))[:max_values])
        for key, values in samples.items()
    ]


def build_metadatas_descript(
    directory: str,
    max_examples_per_key: int = 5,
) -> Dict[str, List[Dict[str, str]]]:
    """Agrège les valeurs trouvées dans les fichiers JSON par clé de métadonnée.

    Retourne un dictionnaire du type ::

        {
            "Metadatas_descript": [
                {
                    "name": "source",
                    "samples": [
                        {"value": "path/to/file.pdf", "example": "Extrait de texte..."},
                        ...
                    ]
                },
                ...
            ]
        }

    :param directory: dossier contenant les JSON à analyser.
    :param max_examples_per_key: nombre d'exemples conservés par clé.
    :return: structure prête à être sérialisée ou envoyée à un LLM.
    """

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {directory}")

    aggregated: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for json_path in sorted(path.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Fichier JSON invalide ignoré: {json_path} ({exc})")
                continue

        for idx, chunk in enumerate(payload):
            metadata = chunk.get("metadata", {}) or {}
            # On récupère un exemple de contenu pour illustrer la provenance
            content = (
                chunk.get("page_content")
                or chunk.get("content")
                or chunk.get("text")
                or ""
            )
            example = " ".join(content.split())[:220] if content else ""

            for key, value in metadata.items():
                samples = aggregated[key]
                if len(samples) >= max_examples_per_key:
                    continue

                entry = {
                    "value": str(value),
                    "source_file": json_path.name,
                }
                if example:
                    entry["example"] = example

                # On évite les doublons exacts pour une même clé
                if entry not in samples:
                    samples.append(entry)

    return {
        "Metadatas_descript": [
            {"name": key, "samples": aggregated[key]}
            for key in sorted(aggregated)
        ]
    }


def build_prompt(samples: Iterable[MetadataSample]) -> str:
    bullet_lines = []
    for sample in samples:
        preview = ", ".join(sample.values)
        bullet_lines.append(f"- {sample.key}: {preview}")

    metadata_block = "\n".join(bullet_lines)

    return f"""Tu es un assistant qui décrit les champs de métadonnées d'une base de documents.
            Les métadonnées observées sont listées ci-dessous avec quelques valeurs exemples :

            {metadata_block}

            Consignes :
            1. Déterminer le type logique de chaque clé (string, integer, float, datetime, boolean, list, dict).
            2. Rédiger une description concise (1 phrase) expliquant la signification du champ.
            3. Répondre exclusivement en JSON valide avec la structure :
            [
            {{"name": "...", "type": "...", "description": "..."}}
            ]

            Le JSON doit contenir toutes les clés observées, rien d'autre.
            Retourne uniquement le JSON comprennant les structures precedante , sans autre texte.
            """


def infer_attribute_info(
    metadata_dir: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: str = "cpu",
) -> List[AttributeInfo]:
    
    directory = Path(metadata_dir)
    samples = load_metadata_samples(directory)
    prompt = build_prompt(samples)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = mod.Mistral7BLLM(
        device=device,
        # pipeline_kwargs={"tokenizer": tokenizer},
        generation_defaults={"max_new_tokens": 700, "temperature": 0.1, "do_sample": False},
    )

    raw_output = llm.generate(prompt)
    print(f"[DEBUG] raw_output:\n\n{raw_output}\n")
    try:
        json_start = raw_output.index("[")
        json_payload = raw_output[json_start:].strip()
        items = json.loads(json_payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Réponse modèle invalide : {raw_output}") from exc

    return [
        AttributeInfo(
            name=item["name"],
            description=item["description"],
            type=item["type"],
        )
        for item in items
    ]


if __name__ == "__main__":
    import utilisation_GPU as test_GPU
    device = test_GPU.test_utilisation_GPU()

    metadata_field_info = infer_attribute_info("src/data/chroma_res",device=device)
    for info in metadata_field_info:
        print(info)
