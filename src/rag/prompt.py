from langchain.prompts import PromptTemplate



class Prompt:
    def __init__(self,index: int = 0):
        self.template = [{ "index" : 0, "prompt" : ( "Use the following pieces of context to answer the question at the end. "
                                                    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                                                    "{context}\n\n"
                                                    "Question: {question}\n"
                                                    "Answer:"
                                                )},
                         { "index" : 1, "prompt" : ( "Tu es un assistant specialise dans l'analyse de documents. "
                                                    "Reponds UNIQUEMENT a partir des informations presentes dans le contexte ci-dessous.\n\n"
                                                    "Regles strictes :\n"
                                                    "- Ne genere JAMAIS d'information qui ne figure pas dans le contexte.\n"
                                                    "- Si le contexte ne contient pas assez d'elements pour repondre, dis-le clairement.\n"
                                                    "- Structure ta reponse avec des paragraphes ou des listes a puces si necessaire.\n"
                                                    "- Cite tes sources entre crochets (ex: [source:nom_du_doc, page X]).\n"
                                                    "- Sois concis et factuel.\n\n"
                                                    "Contexte :\n{context}\n\n"
                                                    "Question : {question}\n\n"
                                                    "Reponse :"
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




