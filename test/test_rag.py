import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import src.modele.modele_LLM_hugface as mod_hug
import src.modele.modele_LLM_ollama as mode_oll
import src.modele.modele_Embeddings as modele_Emb
import src.rag.prompt as prompt
import test.utilisation_GPU as test_GPU
import src.rag.chroma_database as chdt
import src.rag.rag as Rag



if __name__ == "__main__":

    device = test_GPU.test_utilisation_GPU()
    embedder = modele_Emb.Model_embeddings(device,0)
    llm = mode_oll.model_Ollama(2)
    llm_retriever = mode_oll.model_Ollama(2)
    prompt_model = prompt.Prompt(1)
    mode: str = "default"


    rag = Rag.RAG(device,embedder,llm,llm_retriever,prompt_model,mode)

    embed_model = embedder.get_embedder()
    chro_db = chdt.ChromaDB(embed_model)
    embedding_data = chro_db.load()
    print(embedding_data)

    rag.build_data_rag(embedding_data)
    rag.build_pipeline_rag()    
    # rag.chat_with_rag_console("score")
    rag.chat_with_rag_console()