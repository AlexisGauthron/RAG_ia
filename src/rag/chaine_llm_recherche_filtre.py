import sys
import os
from datetime import datetime

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    StructuredQueryOutputParser
)
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.ir import StructuredQuery

import json
import re

import src.rag.parsing as par


# ============================================================
# 1. Exemple de parser personnalisé (tu peux le modifier selon ton besoin)
# ============================================================
from pydantic import Field
from typing import Optional, Set

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    StructuredQuery,
    Operator,
    Operation,
    )
import re


from langchain.chains.query_constructor.base import StructuredQueryOutputParser

class CustomOutputParser(StructuredQueryOutputParser):

    """
    Parseur robuste et paramétrable qui nettoie les filtres invalides (fonctions non reconnues)
    dans la chaîne de filtre d’un JSON généré par le LLM.

    Paramètres :
      - allowed_operators : set des opérateurs/fonctions autorisées.
      - strict : si True, lève une erreur quand fonction inconnue détectée,
                 sinon elle supprime la fonction non reconnue et continue.
    """
    
    """
    Parseur robuste et paramétrable avec Pydantic, pour nettoyer et valider les filtres.
    """

    allowed_functions: Optional[Set[str]] = Field(
        default_factory=lambda: {
            "and", "or", "not", "eq", "ne", "gt", "gte", "lt", "lte",
            "contain", "like", "in", "nin"
        },
        description="Compares ou opérateurs autorisés"
    )
    strict: bool = Field(
        default=False,
        description="Si True, lève une erreur si fonction inconnue détectée"
    )

    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def from_components(
        cls,
        *,
        allowed_functions: Optional[Set[str]] = None,
        strict: bool = False,
        **kwargs,
    ):
        base_parser = super().from_components(**kwargs)
        init_kwargs = {
            "ast_parse": base_parser.ast_parse,
            "strict": strict,
        }
        if allowed_functions is not None:
            init_kwargs["allowed_functions"] = set(allowed_functions)
        return cls(**init_kwargs)
    

    # def parse(self, text: str):
    #     parsed = self.traitement_parse(text)  # ton dict nettoyé
    #     print(f"[DEBUG]  Parsed :  {parsed}\n")
    #     print(f"[DEBUG]  get query Parsed :  {parsed.get("query", "")}\n")
    #     print(f"[DEBUG]  get filter Parsed :  {parsed.get("filter")}\n")
    #     print(f"[DEBUG]  get limit Parsed :  {parsed.get("limit")}\n")

    #     return StructuredQuery(
    #         query=parsed.get("query", ""),
    #         filter=parsed.get("filter"),
    #         limit=parsed.get("limit"),
    #     )


    def parse(self, text: str) -> StructuredQuery:
        data = self.traitement_parse(text)
        print(f"\n[DEBUG] DATA : {data}\n\n")
        cleaned_filter = self.parse_filter_string_to_structured_query(data)

        return cleaned_filter


    def parse_filter_string_to_structured_query(self, parseur) -> StructuredQuery:
        # Exemple simple : on cherche les comparaisons dans la chaîne du filtre au format simple
        # e.g. eq("source", "cours1_Base.pdf")

        query=parseur.get("query", "")
        filter=parseur.get("filter")
        limit=parseur.get("limit")
        
        pattern = re.compile(r'(\w+)\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
        
        comparisons = []
        for match in pattern.finditer(filter):
            comp_str, attr, val = match.groups()

            # Convertit le string du comparateur en enum Comparator, si possible
            try:
                comp_enum = Comparator(comp_str.lower())
            except ValueError:
                raise ValueError(f"Comparateur inconnu : {comp_str}")

            comp = Comparison(
                comparator=comp_enum,
                attribute=attr,
                value=val
            )
            comparisons.append(comp)

        # Combine toutes les comparisons avec un AND
        if len(comparisons) == 1:
            op = comparisons[0]
        elif len(comparisons) > 1:
            op = Operation(operator=Operator.AND, arguments=comparisons)
        else:
            op = None


        print(f"[DEBUG] Comparaisons : {comparisons}")
        # Crée et retourne un StructuredQuery complet
        structured_query = StructuredQuery(
            query=query,
            filter=op,
            limit=limit
        )
        return structured_query





    def traitement_parse(self, text: str):
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if not match:
            raise ValueError(f"[ERROR] Aucun bloc JSON valide trouvé : {text}\n")
        json_str = match.group(1)

        try:
            parsed = par.mega_json_corrector(json_str)
        except Exception as e:
            raise ValueError(f"Erreur JSON lors du parsing : {e}\nContenu :\n{json_str}")

        if "filter" not in parsed or not parsed["filter"]:
            parsed["filter"] = None
            return parsed
        # print(f"[DEBUG]  Parsed entre:  {parsed}\n")
        # parsed["filter"] = self._clean_filter_operator(parsed["filter"])
        # print(f"[DEBUG]  Parsed sortie:  {parsed}\n")
        return parsed






# ============================================================
# 2. CustomSelfQueryRetriever : garde la logique de SelfQueryRetriever
# ============================================================
class CustomSelfQueryRetriever(SelfQueryRetriever):
    """
    Extension du SelfQueryRetriever officiel :
    - ajoute une méthode pour extraire uniquement les filtres JSON
    """

    def extract_filters_json(self, query: str) -> dict:
        """
        Transforme une requête utilisateur en filtre JSON.
        """
        structured_query = self.query_constructor.invoke(query)
        filt = structured_query.filter
        try:
            # Si l'objet Operation du filtre possède une méthode to_dict() :
            return filt.to_dict() if filt else {}
        except AttributeError:
            # Si non, essayer une conversion directe (utile en custom parser)
            return json.loads(filt) if isinstance(filt, str) else {}




# ============================================================
# 3. Fonction principale : build_custom_self_query_retriever
# ============================================================
def build_custom_self_query_retriever(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    allowed_comparators=None,
    allowed_operators=None,
    strict_output_parser=False,      # nouveau paramètre pour contrôle strict du parseur
    enable_limit=True,
    verbose=False,
    test_filtre= True,
):
    
    # Vérification que allowed_operators est bien défini
    if allowed_operators is None:
        raise ValueError("\n[WARM] Le paramètre 'allowed_operators' ne peut pas être None. Veuillez fournir une liste ou un set d'opérateurs autorisés.\n")
    if allowed_comparators is None:
        raise ValueError("\n[WARM] Le paramètre 'allowed_comparators' ne peut pas être None. Veuillez fournir une liste ou un set d'opérateurs autorisés. \n")

    # --- Étape 1 : Création du prompt
    prompt = get_query_constructor_prompt(document_content_description, metadata_field_info)
    if verbose:
        print("\n[DEBUG] Prompt Query Constructor :\n", prompt,'\n\n')

    # --- Étape 2 : Construction du LLM chain
    llm_chain = prompt | llm

    # --- Étape 3 : Création parseur custom avec paramètres passés
    # Si tu veux utiliser StructuredQueryOutputParser, garde cette ligne et configure fix_invalid
    # sinon, instancie CustomOutputParser
    allowed_functions = get_allowed_functions(allowed_comparators, allowed_operators)
    # Usage de CustomOutputParser avec contrôle sur les fonctions autorisées
    custom_output_parser = CustomOutputParser.from_components(
        allowed_functions=set(allowed_functions) if allowed_functions else None,
        strict=strict_output_parser,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
    )

    # --- Étape 4 : Chaîne complète
    query_constructor = llm_chain | custom_output_parser
    if verbose:
        print("\n[DEBUG] Query Constructor configuré avec parser :", type(custom_output_parser).__name__)

    # --- Étape 5 : Création du retriever
    retriever = CustomSelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
        verbose=verbose,
    )

    # try:
    #     test_correction_guillement()
    # except Exception as e:
    #     print("[ERROR] Test correction_parsing",e)

    # --- Étape 6 : Test rapide optionnel
    if test_filtre:
        user_querys = ["Show articles about AAPL with sentiment > 0.8 after January 2024","Parle moi du cours1_Base.pdf, précisément de la page 1 !"]

        for user_query in user_querys:
            raw_llm_output = test_sortie_llm_filtre(user_query,verbose,llm_chain)
            test_parsing_sortie(custom_output_parser,raw_llm_output,verbose)


        # test_creation_filtre(user_query,verbose,retriever)



    return retriever



def test_creation_filtre(user_query,verbose,retriever):
    filters_json = retriever.extract_filters_json(user_query)
    if verbose:
        import json
        print("\n[DEBUG] Filtres détectés (JSON):\n", json.dumps(filters_json, indent=2))


def test_sortie_llm_filtre(user_query,verbose,llm_chain):
    raw_llm_output = llm_chain.invoke(user_query)  # la sortie textuelle

    if verbose:
        print("\n[DEBUG] Sortie brute LLM (avant parsing):\n", raw_llm_output)

    return raw_llm_output


def test_parsing_sortie(custom_output_parser,raw_llm_output,verbose):
    parsed_output = custom_output_parser.parse(raw_llm_output)
    if verbose:
        print("\n[DEBUG] Sortie après parsing :\n", parsed_output)






def get_allowed_functions(comparators, operators):
    functions = []

    # Ajoute les comparateurs en minuscules
    for comp in comparators:
        # Si c'est une énumération, utilise .name ou .value
        if hasattr(comp, 'name'):
            functions.append(comp.name.lower())
        else:
            functions.append(str(comp).lower())

    # Ajoute les opérateurs en minuscules
    for op in operators:
        if hasattr(op, 'name'):
            functions.append(op.name.lower())
        else:
            functions.append(str(op).lower())

    return functions







