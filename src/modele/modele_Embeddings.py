from langchain_huggingface import HuggingFaceEmbeddings

modele_embedding = [{"index" : 0, "model" : "sentence-transformers/all-MiniLM-L6-v2"},
                    {"index" : 1, "model" : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}]

class Model_embeddings:
    def __init__(self, device, index: int = 0):
        self.index = index
        self.model = modele_embedding[index]["model"]
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={"device": device}
        ) 
        print("[INFO] Chargement Model_embeddings\n")
    
    def get_embedder(self):
        return self.embedder
    



    