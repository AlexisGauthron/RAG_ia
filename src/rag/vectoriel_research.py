import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import json
from typing import Optional, Dict, Any
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.ir import Comparator, Operator

import src.rag.chaine_llm_recherche_filtre as ch_filtre


class Vectoriel_research:
    def __init__(self, vector_db=None):
        """
        :param device: 0 pour GPU, -1 pour CPU
        :param embed_model: Modèle d'embedding HF
        """
        self.vectordb = vector_db
        self.metadata_field_info = None
        self.retriever = None
        self.document_content_description = None
        self.allowed_comparators = None
        self.allowed_operators = None   


    def search(self, top_k: int = 5):
        if not self.vectordb:
            print("[WARN] L'index n'est pas construit.")
            return []
        self.retriever = self.vectordb.as_retriever(search_kwargs={
            "k": top_k,                  # top 6 docs 
            # "fetch_k": 20,           # pré-sélection de 20 chunks
            # "score_threshold": 0.7,  # filtrage des docs peu similaires
            # "search_type": "similarity",     # ou "mmr"
            # "lambda_mult": 0.5,      # pondération (pour recherche hybride)
        })


    def build_metadata_info(self):
        from langchain.retrievers.self_query.base import AttributeInfo

        self.metadata_field_info = [
            AttributeInfo(
                name="producer",
                description="Logiciel ou moteur ayant produit le PDF final.",
                type="string",
            ),
            AttributeInfo(
                name="creator",
                description="Application utilisée pour créer le document d’origine.",
                type="string",
            ),
            AttributeInfo(
                name="creationdate",
                description="Horodatage ISO de création du document.",
                type="datetime",
            ),
            AttributeInfo(
                name="moddate",
                description="Horodatage ISO de la dernière modification du document.",
                type="datetime",
            ),
            AttributeInfo(
                name="trapped",
                description="Indique si le document PDF est marqué comme 'trapped' (True/False).",
                type="string",
            ),
            AttributeInfo(
                name="source",
                description="Chemin absolu du fichier source d’où provient le contenu.",
                type="string",
            ),
            AttributeInfo(
                name="total_pages",
                description="Nombre total de pages indiqué dans le document.",
                type="integer",
            ),
            AttributeInfo(
                name="page",
                description="Indice de la page courante (base zéro) pour ce chunk.",
                type="integer",
            ),
            AttributeInfo(
                name="page_label",
                description="Numérotation affichée sur la page telle qu’affichée dans le PDF.",
                type="string",
            ),
            AttributeInfo(
                name="author",
                description="Auteur du document, si disponible.",
                type="string",
            ),
            AttributeInfo(
                name="extension",
                description="Extension du fichier (.pdf, etc..)",
                type="string",
            ),
        ]
        
        self.document_content_description = "Texte extrait de documents divers (PDF, Word, etc.)."

        self.allowed_comparators=[
                Comparator.EQ,   # eq
                Comparator.NE,   # ne
                Comparator.GT,   # gt
                Comparator.GTE,  # gte
                Comparator.LT,   # lt
                Comparator.LTE,  # lte
            ],
        self.allowed_operators=[
                Operator.AND,    # and
                Operator.OR,     # or
                # (Operator.NOT existe, mais tu n'en as pas besoin selon ton schéma)
            ],


    def search_llm(self, llm, methode: str = "default1"):
        
        # Definit les métadonnées et descriptions
        self.build_metadata_info()
        print("\n[DEBUG] Metadata info:", self.metadata_field_info)

        if methode == "default":
            print("\n[INFO] Initialisation du retriever auto-filtrant (SelfQueryRetriever) avec la méthode par défaut.\n")
            self.retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=self.vectordb,
                document_contents=self.document_content_description,   # description courte du contenu
                metadata_field_info=self.metadata_field_info,          # tes champs filtrables
                allowed_comparators=self.allowed_comparators,
                allowed_operators=self.allowed_operators,
                enable_limit=True,   # permet: "donne-moi 3 docs"
                verbose=True,
            )
        else:
            print("\n[INFO] Initialisation du retriever auto-filtrant (SelfQueryRetriever) avec une méthode personnalisée.\n")



            self.retriever = ch_filtre.build_custom_self_query_retriever(
                llm=llm,
                vectorstore=self.vectordb,
                document_content_description=self.document_content_description,
                metadata_field_info=self.metadata_field_info,
                allowed_comparators=self.allowed_comparators,
                allowed_operators=self.allowed_operators,
                enable_limit=True,
                verbose=True,
            )


        print("[INFO] Retriever LLM initialisé.\n\n self.retriever:", self.retriever)

        return self.retriever




    def get_retriever(self):
        return self.retriever
    











def corriger_sortie_llm(texte_llm: str) -> Optional[Dict[str, Any]]:
    """
    Corrige et structure la sortie du LLM si elle contient les champs attendus :
    "query", "filter" et éventuellement "limit".

    Paramètres :
    - texte_llm : str → la sortie textuelle du LLM (potentiellement du JSON entouré de texte).

    Retour :
    - dict structuré avec les bonnes clés, ou None si le parsing échoue.
    """
    try:
        # 🧹 1. Extraire le JSON du texte brut
        debut_json = texte_llm.find("{")
        fin_json = texte_llm.rfind("}") + 1
        texte_json = texte_llm[debut_json:fin_json]

        # 📦 2. Parser le JSON
        data = json.loads(texte_json)

        # ✅ 3. Valider les champs obligatoires
        if "query" in data and "filter" in data:
            # Optionnel : filtrer les champs attendus uniquement
            resultat = {
                "query": data["query"],
                "filter": data["filter"],
            }
            if "limit" in data:
                resultat["limit"] = data["limit"]
            return resultat

        # ❌ Champs manquants
        print("[ERREUR] Champs obligatoires manquants dans la réponse LLM.")
        return None

    except Exception as e:
        print(f"[ERREUR] Échec du parsing JSON de la sortie LLM : {e}")
        return None


        






