import json
from typing import Optional, Dict, Any
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.ir import Comparator, Operator
from langchain.chains import LLMChain
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    StructuredQueryOutputParser,
    load_query_constructor_runnable
)



def build_custom_self_query_retriever(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    allowed_comparators=None,
    allowed_operators=None,
    enable_limit=True,
    verbose=False,
):
    # 1. Obtenir le prompt par défaut
    prompt = get_query_constructor_prompt(document_content_description, metadata_field_info)
    if verbose:
        print("\n[DEBUG] Prompt Query Constructor :\n", prompt)

    # 2. Créer le LLMChain (dans les versions récentes, utilise plutôt le pipe)
    llm_chain = load_query_constructor_runnable(
        llm,
        document_content_description,
        metadata_field_info
    )

    # 3. Créer un OutputParser personnalisé
    custom_output_parser = CustomOutputParser.from_components()

    # 4. Créer la chaîne query_constructor avec le parseur personnalisé
    query_constructor = llm_chain | custom_output_parser

    # 5. Construire le SelfQueryRetriever avec les paramètres transmis
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
        verbose=verbose,
    )

    return retriever




class CustomOutputParser(StructuredQueryOutputParser):
        def parse(self, text: str):
            print("\n\n ####################### [DEBUG] Réponse brute LLM ###################\n", text)  # affiche la sortie textuelle brute
            # Correction simple sur la sortie brute LLM
            text = text.replace(" and ", " AND ")
            # Appeler le parseur parent pour parser correctement
            return super().parse(text)