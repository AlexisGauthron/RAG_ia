import streamlit as st
from streamlit.web import cli as stcli
import sys
import os
import glob




import load_fichier as gf
from load_fichier import chemindossier
CHEMIN_FICHIER = chemindossier()

import embedding as emb
all_extension = gf.LISTE_FICHIER_ACCEPTE
extension = gf.LISTE_ACTUEL


import module_app as mapp

app = mapp.module_app(embed_model=0, prompt_model=1, directory=CHEMIN_FICHIER)

CHEMIN_DOSSIER_RAG = f"{CHEMIN_FICHIER}/data_rag"



### Debut de l'application Streamlit ###


st.title('Bonjour, bienvenue sur mon RAG!')


def upload_fichier():

    uploaded_files = st.file_uploader("Ajoutez votre fichier", type=extension, accept_multiple_files=True)

    if uploaded_files:

        # Formulaire pour confirmer l’upload
        with st.form("form_confirm_upload"):
            submit_button = st.form_submit_button("Confirmer le chargement")

            if submit_button:
                nom_subdir = "Importer"
                nom_entier = f"{CHEMIN_FICHIER}/{nom_subdir}"
                  # Vous pouvez modifier cela selon vos besoins
                saved_paths = [gf.save_uploaded_file(f, subdir=nom_subdir, dossier=CHEMIN_FICHIER) for f in uploaded_files]
                st.success(f"{len(saved_paths)} fichier(s) enregistré(s) dans: {CHEMIN_FICHIER}/{nom_subdir}")
                for p in saved_paths:
                    st.write(str(p))
                # (Optionnel) garder les chemins pour usage ultérieur
                st.session_state["uploaded_paths"] = [str(p) for p in saved_paths]
                app.telechargement(nom_entier)
        
        
        
def discuter():
    query = st.text_input('Poser vos questions !')
    if query:
        app.lancement_RAG(llm_model=0, llm_retriever_model=0)
        response = app.question_reponse_rag(query)
        st.write(f"{response["result"]}")


def sidebar(Liste_document_RAG):

    # Sidebar avec un selectbox (menu dépliant)

    
    with st.sidebar:
        st.header("Documents")  # Titre dans la sidebar
        

        upload_fichier()


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

        fichiers = [
            {"nom": "GUIDE_ULTIME_INVESTISSEMENT-IMMOBILIER_DUR.pdf", "taille": "4.8MB"},
            {"nom": "MON_DOCUMENT_TRES_LONG_OU_LA_LIGNE_DOIT_ETRE_COUPEE_AUTOMATIQUEMENT.pdf", "taille": "9.1MB"}
        ]

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
                        st.sidebar.write(f"Suppression simulée : {fichier}")










all_files_rag = gf.find_all_files(CHEMIN_DOSSIER_RAG)


discuter()

sidebar(all_files_rag)



# if __name__ == "__main__":
#     sys.argv = ["streamlit", "run", "src/front/app.py"]  # remplacer par votre fichier
#     sys.exit(stcli.main())
