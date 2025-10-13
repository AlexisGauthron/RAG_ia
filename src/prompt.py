from langchain.prompts import PromptTemplate



class Prompt:
    def __init__(self,index: int = 0):
        self.template = [{ "index" : 0, "prompt" : ( "Use the following pieces of context to answer the question at the end. "
                                                    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                                                    "{context}\n\n"
                                                    "Question: {question}\n"
                                                    "Answer:"
                                                )},
                         { "index" : 1, "prompt" : ( "Utilise les informations suivantes pour répondre à la question posée.\n"
                                                    "Si tu ne connais pas la réponse, dis simplement que tu ne sais pas, n'invente pas une réponse.\n\n"
                                                    "{context}\n\n"
                                                    "Question : {question}\n"
                                                    "Réponds de façon détaillée et cite les sources (ex: [source:nom_du_doc])."
                                                )},
                         { "index" : 2, "prompt" : ( """Tu es un assistant. Pour chaque passage référencé en réponse, cite la source en mentionnant le nom ou la position du chunk/document (voir contexte).
                                                    Question : {question}
                                                    Contexte : {context}
                                                    Réponds de façon détaillée et cite les sources (ex: [source:nom_du_doc])."""
                                                )}
                                                
                                                ]



        self.prompt = PromptTemplate(
            template=self.template[index]["prompt"],
            input_variables=["context", "question"]  # adapte aux {…} présents dans ta template
        )

    def get_prompt(self):
        return self.prompt




