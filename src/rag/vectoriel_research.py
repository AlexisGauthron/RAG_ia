import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import json
from typing import Optional, Dict, Any, List
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.ir import Comparator, Operator
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

import src.rag.chaine_llm_recherche_filtre as ch_filtre


class FallbackRetriever(BaseRetriever):
    """Retriever avec fallback : si le primary echoue ou retourne 0 resultats, utilise le fallback."""
    primary_retriever: BaseRetriever
    fallback_retriever: BaseRetriever

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        try:
            docs = self.primary_retriever.invoke(query)
            if docs:
                return docs
            print("[INFO] Fallback MMR active : le filtre n'a retourne aucun resultat")
            return self.fallback_retriever.invoke(query)
        except Exception as e:
            print(f"[INFO] Fallback MMR active : erreur du filtre ({e})")
            return self.fallback_retriever.invoke(query)


class Vectoriel_research:
    def __init__(self, vector_db=None, embedder=None):
        """
        :param vector_db: Base vectorielle ChromaDB
        :param embedder: Modele d'embedding pour le filtrage par score
        """
        self.vectordb = vector_db
        self.embedder = embedder
        self.metadata_field_info = None
        self.retriever = None
        self.document_content_description = None
        self.allowed_comparators = None
        self.allowed_operators = None


    def search(self, top_k: int = 6, llm=None):
        if not self.vectordb:
            print("[WARN] L'index n'est pas construit.")
            return []

        print(f"[INFO] Recherche MMR activee (k={top_k}, fetch_k=20, lambda_mult=0.7)")
        base_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            }
        )

        # Filtrage par seuil de pertinence via EmbeddingsFilter
        if self.embedder:
            print("[INFO] EmbeddingsFilter active (seuil=0.35)")
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embedder,
                similarity_threshold=0.35
            )
            base_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=base_retriever
            )

        # Expansion de requete via MultiQueryRetriever (mode default uniquement)
        if llm:
            print("[INFO] MultiQueryRetriever active (3 variantes)")
            base_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )

        self.retriever = base_retriever


    def build_metadata_info(self, available_sources=None):
        from langchain.retrievers.self_query.base import AttributeInfo

        source_description = "Nom du fichier PDF source (ex: ‘Cours1_Base-8cd6a096.pdf’)."
        if available_sources:
            source_description = f"Nom du fichier source. Valeurs disponibles : {available_sources}"

        self.metadata_field_info = [
            AttributeInfo(
                name="source",
                description=source_description,
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="Numero de page (base zero) du chunk dans le document (ex: 0, 1, 2...).",
                type="integer",
            ),
            AttributeInfo(
                name="total_pages",
                description="Nombre total de pages du document (ex: 12, 45).",
                type="integer",
            ),
            AttributeInfo(
                name="author",
                description="Auteur du document, si disponible (ex: ‘Jean Dupont’).",
                type="string",
            ),
            AttributeInfo(
                name="page_label",
                description="Numerotation affichee sur la page dans le PDF (ex: ‘1’, ‘2’, ‘iii’).",
                type="string",
            ),
            AttributeInfo(
                name="creationdate",
                description="Date de creation du document au format ISO (ex: ‘D:20240115’).",
                type="datetime",
            ),
            AttributeInfo(
                name="moddate",
                description="Date de derniere modification au format ISO (ex: ‘D:20240220’).",
                type="datetime",
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


    def search_llm(self, llm, methode: str = "default1", top_k: int = 6):

        # Extraire les sources disponibles depuis ChromaDB
        available_sources = []
        try:
            all_data = self.vectordb._collection.get(include=["metadatas"])
            available_sources = sorted(set(
                os.path.basename(m.get("source", ""))
                for m in all_data.get("metadatas", []) if m.get("source")
            ))
            print(f"[INFO] Sources disponibles : {available_sources}")
        except Exception as e:
            print(f"[WARN] Impossible d'extraire les sources : {e}")

        # Definit les metadonnees et descriptions
        self.build_metadata_info(available_sources=available_sources)
        print("\n[DEBUG] Metadata info:", self.metadata_field_info)

        if methode == "default":
            print("\n[INFO] Initialisation du retriever auto-filtrant (SelfQueryRetriever) avec la methode par defaut.\n")
            self.retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=self.vectordb,
                document_contents=self.document_content_description,
                metadata_field_info=self.metadata_field_info,
                allowed_comparators=self.allowed_comparators,
                allowed_operators=self.allowed_operators,
                enable_limit=True,
                verbose=True,
            )
        else:
            print("\n[INFO] Initialisation du retriever auto-filtrant (SelfQueryRetriever) avec methode personnalisee + MMR.\n")

            # Construire le SelfQueryRetriever avec MMR
            filter_retriever = ch_filtre.build_custom_self_query_retriever(
                llm=llm,
                vectorstore=self.vectordb,
                document_content_description=self.document_content_description,
                metadata_field_info=self.metadata_field_info,
                allowed_comparators=self.allowed_comparators,
                allowed_operators=self.allowed_operators,
                strict_output_parser=False,
                enable_limit=True,
                verbose=True,
                search_type="mmr",
                search_kwargs={"fetch_k": 20, "lambda_mult": 0.7, "k": top_k},
                available_sources=available_sources,
            )
            print(f"[INFO] MMR active sur SelfQueryRetriever (k={top_k}, fetch_k=20, lambda_mult=0.7)")

            # Wrap avec EmbeddingsFilter pour reranking (seuil bas car le filtre
            # metadata source cible deja le bon document)
            if self.embedder:
                print("[INFO] EmbeddingsFilter active sur SelfQueryRetriever (seuil=0.15)")
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embedder,
                    similarity_threshold=0.15
                )
                filter_retriever = ContextualCompressionRetriever(
                    base_compressor=embeddings_filter,
                    base_retriever=filter_retriever
                )

            # Construire le fallback MMR (sans filtre LLM)
            print(f"[INFO] Construction du fallback MMR (k={top_k})")
            fallback_retriever = self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": 20,
                    "lambda_mult": 0.7,
                }
            )
            if self.embedder:
                fallback_embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embedder,
                    similarity_threshold=0.15
                )
                fallback_retriever = ContextualCompressionRetriever(
                    base_compressor=fallback_embeddings_filter,
                    base_retriever=fallback_retriever
                )

            # Wrapper avec FallbackRetriever
            self.retriever = FallbackRetriever(
                primary_retriever=filter_retriever,
                fallback_retriever=fallback_retriever
            )

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


        






