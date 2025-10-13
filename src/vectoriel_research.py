import re, json

CODEFENCE_JSON_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)
CODEFENCE_ANY_RE = re.compile(
    r"```\s*(\{.*?\})\s*```", re.DOTALL
)

def extract_json_codeblock(text: str) -> str | None:
    """
    Retourne la *chaîne JSON* à l'intérieur d'un bloc ```json ... ``` si présent.
    Si pas de bloc avec `json`, tente un bloc ``` ... ``` générique contenant {...}.
    Sinon renvoie None.
    """
    m = CODEFENCE_JSON_RE.search(text)
    if m:
        return m.group(1).strip()
    m = CODEFENCE_ANY_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


class Vectoriel_research:
    def __init__(self, vector_db=None):
        """
        :param device: 0 pour GPU, -1 pour CPU
        :param embed_model: Modèle d'embedding HF
        """
        self.vectordb = vector_db
        self.retriever = None
    

    def search(self, top_k: int = 5):
        if not self.vectordb:
            print("[WARN] L'index n'est pas construit.")
            return []
        self.retriever = self.vectordb.as_retriever(search_kwargs={
            "k": 6,                  # top 6 docs 
            "fetch_k": 20,           # pré-sélection de 20 chunks
            "score_threshold": 0.7,  # filtrage des docs peu similaires
            "filter": {"author": "Alexis"},  # ex: filtres personnalisés
            "search_type": "similarity",     # ou "mmr"
            "lambda_mult": 0.5,      # pondération (pour recherche hybride)
        })

    
    def search_llm(self, llm):
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain.chains.query_constructor.base import AttributeInfo

        # --- 1️⃣ Définition des métadonnées filtrables ---
        metadata_field_info = [
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
        ]

        # --- 2️⃣ Description du contenu global ---
        document_content_description = "Texte extrait de documents divers (PDF, Word, etc.)."

        from langchain.chains.query_constructor.ir import Comparator, Operator

        # --- 3️⃣ Création du retriever auto-filtrant avec opérateurs/comparateurs autorisés ---
        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=self.vectordb,
            document_contents=document_content_description,   # description courte du contenu
            metadata_field_info=metadata_field_info,          # tes champs filtrables
            allowed_comparators=[
                Comparator.EQ,   # eq
                Comparator.NE,   # ne
                Comparator.GT,   # gt
                Comparator.GTE,  # gte
                Comparator.LT,   # lt
                Comparator.LTE,  # lte
            ],
            allowed_operators=[
                Operator.AND,    # and
                Operator.OR,     # or
                # (Operator.NOT existe, mais tu n'en as pas besoin selon ton schéma)
            ],
            enable_limit=True,   # permet: "donne-moi 3 docs"
            verbose=True,
        )
        self.retriever = extract_json_codeblock(self.retriever)

        print("[INFO] Retriever LLM initialisé.\n\n self.retriever:", self.retriever)

        return self.retriever



    def get_retriever(self):
        return self.retriever
    


        






