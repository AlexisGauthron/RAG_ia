from langchain_huggingface import HuggingFaceEmbeddings

class Model_embeddings:
    def __init__(self, device, modele : str):
        self.model = modele
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={"device": device}
        ) 
        print("[INFO] Chargement Model_embeddings\n")
    
    def get_embedder(self):
        return self.embedder
    



    