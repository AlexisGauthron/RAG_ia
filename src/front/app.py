import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)



import streamlit as st
from streamlit.web import cli as stcli
import glob




import src.rag.load_fichier as lf
import src.gestionnaire_fichier as gf
from src.gestionnaire_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()

import src.rag.embedding as emb
all_extension = lf.LISTE_FICHIER_ACCEPTE
extension = lf.LISTE_ACTUEL


CHEMIN_FICHIER_RAG = f"{CHEMIN_FICHIER}/data_rag"
import src.front.module_app as mapp

import src.modele.modele_LLM_ollama as mode_oll

### Debut de l'application Streamlit ###
class App():
    def __init__(self):
        
        self.app = mapp.module_app(embed_model=0, prompt_model=1, directory=CHEMIN_FICHIER_RAG)
        self.data_rag = gf.find_all_files(CHEMIN_FICHIER_RAG)
        pass

    
    def titre(self):
        st.title('Bonjour, bienvenue sur mon RAG!')

    def take_all_files(self):
        self.data_rag = gf.find_all_files(CHEMIN_FICHIER_RAG)

    def upload_fichier(self,name_key,):

        uploaded_files = st.file_uploader("Ajoutez votre fichier", type=extension, accept_multiple_files=True,key=name_key)

        if uploaded_files:

            # Formulaire pour confirmer l’upload
            with st.form("form_confirm_upload"):
                submit_button = st.form_submit_button("Confirmer le chargement")

                if submit_button:
                    nom_subdir = "Importer"
                    nom_entier = f"{CHEMIN_FICHIER}/{nom_subdir}"
                    # Vous pouvez modifier cela selon vos besoins
                    saved_paths = [lf.save_uploaded_file(f, subdir=nom_subdir, dossier=CHEMIN_FICHIER) for f in uploaded_files]
                    for p in saved_paths:
                        st.write(str(p))
                    # (Optionnel) garder les chemins pour usage ultérieur
                    # st.session_state["uploaded_paths"] = [str(p) for p in saved_paths]
                    doublons = self.app.telechargement(nom_entier)
                    if doublons == 1:
                        message = f"{len(saved_paths)} fichiers enregistré cependant suppression doublons"
                        st.markdown(
                            f"""
                            <div style="
                                padding: 0.75rem 1rem;
                                border-radius: 0.5rem;
                                background-color: #ffa726;
                                color: #1f1f1f;
                                font-weight: 600;
                            ">
                                {message}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.success(f"{len(saved_paths)} fichiers enregistré")

        

    def discuter(self):

        # "Mémoire" pour garder historique du chat
        if "dialogue" not in st.session_state:
            st.session_state.dialogue = []

        # Barre d'input sous forme de formulaire pour UX plus nette
        with st.form("form_question"):
            query = st.text_input('Posez votre question...')
            submit = st.form_submit_button('Envoyer')

        # Gestion de la réponse si bouton cliqué et input non vide
        if submit and query:
            self.app.lancement_RAG(mode_oll.model_Ollama(1), mode_oll.model_Ollama(1))
            response = self.app.question_reponse_rag(query)
            if response is not None:
                # Ajout question/réponse à l'historique
                st.session_state.dialogue.append({"question": query, "réponse": response["result"]})

        # Affichage historique du chat
        for i, turn in enumerate(st.session_state.dialogue):
            st.markdown(f"**Vous :** {turn['question']}")
            st.write(f"**Chatbot :** {turn['réponse']}")




    def sidebar(self):

        # Sidebar avec un selectbox (menu dépliant)
        fichiers = gf.find_all_files(CHEMIN_FICHIER_RAG)
        
        with st.sidebar:
            st.header("Documents")  # Titre dans la sidebar
            

            self.upload_fichier("A")


            # Injecter le CSS
            st.markdown(
                """
                <style>
                /* Supprimer la bordure autour des colonnes contenant les fichiers */
                .files-row {
                    border: none !important;
                    padding: 0 !important;
                    margin-bottom: 5px;
                }

                /* Afficher le nom du fichier sur une seule ligne avec ellipsis */
                .file-name {
                    display: -webkit-box;
                    -webkit-box-orient: vertical;
                    -webkit-line-clamp: 2;       /* Limite à 2 lignes */
                    overflow: hidden;
                    font-size: 12px !important;  /* Police réduite */
                    max-width: 180px;
                    line-height: 1.15;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # fichiers = [
            #     {"nom": "GUIDE_ULTIME_INVESTISSEMENT-IMMOBILIER_DUR.pdf", "taille": "4.8MB"},
            #     {"nom": "MON_DOCUMENT_TRES_LONG_OU_LA_LIGNE_DOIT_ETRE_COUPEE_AUTOMATIQUEMENT.pdf", "taille": "9.1MB"}
            # ]

            with st.sidebar.expander("Vos Documents existant"):
                for i, fichier in enumerate(fichiers):
                    cols = st.columns([0.1, 0.7, 0.2], gap="small")

                    # Ajouter une classe CSS personnalisée au container en HTML
                    with cols[0]:
                        st.markdown(f'<div class="files-row">📄</div>', unsafe_allow_html=True)

                    with cols[1]:
                        st.markdown(f'<span class="file-name">{fichier}</span>', unsafe_allow_html=True)

                    with cols[2]:
                        if st.button("X", key=f"del_{i}"):
                            self.app.delete_files(fichier)
                            fichier = gf.find_all_files(CHEMIN_FICHIER_RAG)
                            st.rerun()


    def template(self):
        self.titre()
        self.discuter()
        self.sidebar()








app = App()
app.template()




# if __name__ == "__main__":
#     sys.argv = ["streamlit", "run", "src/front/app.py"]  # remplacer par votre fichier
#     sys.exit(stcli.main())
