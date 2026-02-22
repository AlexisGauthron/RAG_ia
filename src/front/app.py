import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import streamlit as st
from streamlit.web import cli as stcli
import src.rag.prompt as prompt

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

modele_embedding = [
    {"index": 0, "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"index": 1, "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
]

model_ollama = ["llama3.2:3b", "llama3.2:1b", "mistral:7b-instruct", "deepseek-r1:8b"]

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalise ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ═══════════════════════════════════════════════
       VARIABLES & RESET
       ═══════════════════════════════════════════════ */
    :root {
        --accent: #7C6BFF;
        --accent-light: #A99BFF;
        --accent-bg: rgba(124, 107, 255, 0.08);
        --accent-bg-hover: rgba(124, 107, 255, 0.15);
        --success: #34D399;
        --danger: #F87171;
        --glass: rgba(255, 255, 255, 0.04);
        --glass-border: rgba(255, 255, 255, 0.08);
        --radius: 14px;
        --radius-sm: 10px;
    }

    /* ═══════════════════════════════════════════════
       GLOBAL — centrage du contenu
       ═══════════════════════════════════════════════ */
    .main .block-container {
        max-width: 820px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
    }

    /* ═══════════════════════════════════════════════
       SIDEBAR FIXE — masquer les boutons replier/ouvrir
       ═══════════════════════════════════════════════ */
    /* Bouton fleche de repli dans la sidebar */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }
    /* Bouton >> pour rouvrir la sidebar (quand repliee) */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Forcer la sidebar a rester visible */
    section[data-testid="stSidebar"] {
        transform: none !important;
        transition: none !important;
    }

    /* ═══════════════════════════════════════════════
       ANIMATED HERO HEADER
       ═══════════════════════════════════════════════ */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }

    .hero-wrapper {
        text-align: center;
        padding: 2.2rem 1.5rem 1.6rem;
        border-radius: var(--radius);
        background: linear-gradient(135deg, rgba(124,107,255,0.12), rgba(56,189,248,0.10), rgba(168,85,247,0.10));
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border: 1px solid var(--glass-border);
        margin-bottom: 1.5rem;
    }
    .hero-icon {
        font-size: 2.8rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    .hero-wrapper h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0 0.15rem;
        background: linear-gradient(135deg, var(--accent), #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInUp 0.6s ease both;
    }
    .hero-wrapper .hero-sub {
        opacity: 0.6;
        font-size: 0.95rem;
        margin: 0;
        animation: fadeInUp 0.6s ease 0.15s both;
    }

    /* ═══════════════════════════════════════════════
       CHAT SECTION HEADER
       ═══════════════════════════════════════════════ */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .chat-header-icon {
        font-size: 1.15rem;
    }
    .chat-header-text {
        font-size: 1.05rem;
        font-weight: 700;
    }

    /* ═══════════════════════════════════════════════
       CHAT BUBBLES
       ═══════════════════════════════════════════════ */
    div[data-testid="stChatMessage"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(6px);
        animation: fadeInUp 0.35s ease both;
        padding: 0.85rem 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* ═══════════════════════════════════════════════
       FILTER BADGE
       ═══════════════════════════════════════════════ */
    .filter-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.65rem;
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-left: 0.6rem;
        vertical-align: middle;
        transition: all 0.2s;
    }
    .filter-on {
        background: rgba(52, 211, 153, 0.14);
        color: var(--success);
        box-shadow: 0 0 8px rgba(52, 211, 153, 0.15);
    }
    .filter-off {
        background: rgba(248, 113, 113, 0.10);
        color: var(--danger);
    }

    /* ═══════════════════════════════════════════════
       SOURCE CARDS
       ═══════════════════════════════════════════════ */
    .source-card {
        border-left: 3px solid var(--accent);
        padding: 0.7rem 1rem;
        margin-bottom: 0.55rem;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        background: var(--accent-bg);
        transition: background 0.2s, transform 0.2s;
    }
    .source-card:hover {
        background: var(--accent-bg-hover);
        transform: translateX(3px);
    }
    .source-card .source-meta {
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: var(--accent-light);
        font-size: 0.78rem;
        letter-spacing: 0.01em;
    }
    .source-card .source-text {
        white-space: pre-wrap;
        opacity: 0.8;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* ═══════════════════════════════════════════════
       EMPTY STATE
       ═══════════════════════════════════════════════ */
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50%      { opacity: 1; }
    }
    .empty-chat {
        text-align: center;
        padding: 3rem 1rem;
        opacity: 0.55;
        animation: fadeInUp 0.5s ease both;
    }
    .empty-chat .empty-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: pulse 2.5s ease-in-out infinite;
    }
    .empty-chat p {
        font-size: 0.95rem;
        margin: 0.2rem 0;
    }

    /* ═══════════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        min-width: 310px;
        max-width: 370px;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    /* Sidebar title */
    .sidebar-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.9rem;
    }
    .sidebar-title-icon {
        font-size: 1.35rem;
    }

    /* Stat pills */
    .stat-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.1rem;
    }
    .stat-pill {
        flex: 1;
        text-align: center;
        padding: 0.6rem 0.4rem;
        border-radius: var(--radius-sm);
        background: var(--accent-bg);
        border: 1px solid var(--glass-border);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 107, 255, 0.12);
    }
    .stat-pill .stat-num {
        font-size: 1.4rem;
        font-weight: 800;
        display: block;
        background: linear-gradient(135deg, var(--accent), #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-pill .stat-label {
        font-size: 0.72rem;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* File cards */
    .file-card {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.8rem;
        border-radius: var(--radius-sm);
        background: var(--glass);
        border: 1px solid var(--glass-border);
        margin-bottom: 0.35rem;
        transition: all 0.2s;
    }
    .file-card:hover {
        background: var(--accent-bg);
        border-color: rgba(124, 107, 255, 0.2);
    }
    .file-icon {
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .file-name {
        flex: 1;
        font-size: 0.82rem;
        line-height: 1.3;
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        word-break: break-all;
    }

    /* Delete button alignment */
    .del-btn-wrapper {
        display: flex;
        align-items: center;
        height: 100%;
        padding-top: 0.3rem;
    }

    /* Upload section */
    .upload-section {
        margin-bottom: 0.8rem;
    }

    /* Section divider label */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.45;
        font-weight: 700;
        margin-bottom: 0.5rem;
        margin-top: 0.2rem;
    }

    /* ═══════════════════════════════════════════════
       SCROLLBAR (subtle)
       ═══════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(124, 107, 255, 0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 107, 255, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


### Debut de l'application Streamlit ###
class App:
    def __init__(self):
        self.app = mapp.module_app(
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            prompt_model=prompt.Prompt(1),
            directory=CHEMIN_FICHIER_RAG,
        )
        self.data_rag = gf.find_all_files(CHEMIN_FICHIER_RAG)
        # Sync au premier lancement uniquement
        if not st.session_state.get("sync_done"):
            self.app.sync_files()
            st.session_state.sync_done = True

    # ─── Header ──────────────────────────────────────────────────────────────
    def titre(self):
        st.markdown(
            """
            <div class="hero-wrapper">
                <span class="hero-icon">🤖</span>
                <h1>RAG Assistant</h1>
                <p class="hero-sub">Interrogez vos documents en langage naturel &mdash; propulse par Ollama</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def take_all_files(self):
        self.data_rag = gf.find_all_files(CHEMIN_FICHIER_RAG)

    # ─── Upload fichier ─────────────────────────────────────────────────────
    def upload_fichier(self, name_key):
        uploaded_files = st.file_uploader(
            "Glissez vos fichiers ici",
            type=extension,
            accept_multiple_files=True,
            key=name_key,
        )

        if uploaded_files:
            with st.form("form_confirm_upload"):
                st.caption(f"{len(uploaded_files)} fichier(s) selectionne(s)")
                submit_button = st.form_submit_button(
                    "Confirmer le chargement",
                    use_container_width=True,
                    type="primary",
                )

                if submit_button:
                    nom_subdir = "Importer"
                    saved_paths = [
                        lf.save_uploaded_file(f, subdir=nom_subdir, dossier=CHEMIN_FICHIER)
                        for f in uploaded_files
                    ]

                    doublons = self.app.telechargement()
                    st.session_state.rag_initialized = False
                    if doublons == 1:
                        st.warning(
                            f"{len(saved_paths)} fichier(s) enregistre(s) — doublons supprimes",
                            icon="⚠️",
                        )
                    else:
                        st.success(
                            f"{len(saved_paths)} fichier(s) enregistre(s) avec succes",
                            icon="✅",
                        )

    # ─── Chat principal ──────────────────────────────────────────────────────
    def discuter(self):
        st.session_state.setdefault("dialogue", [])
        st.session_state.setdefault("filtre_actif", False)
        st.session_state.setdefault("rag_mode", None)
        st.session_state.setdefault("rag_initialized", False)
        st.session_state.setdefault("top_k", 6)

        # Header de la section chat
        col_title, col_filter = st.columns([5, 1])
        with col_title:
            st.markdown(
                '<div class="chat-header">'
                '<span class="chat-header-icon">💬</span>'
                '<span class="chat-header-text">Conversation</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_filter:
            st.session_state.filtre_actif = st.toggle(
                "Filtre", value=st.session_state.filtre_actif
            )

        # Etat vide — message d'accueil
        if not st.session_state.dialogue:
            st.markdown(
                """
                <div class="empty-chat">
                    <span class="empty-icon">💬</span>
                    <p><strong>Aucune conversation pour l'instant</strong></p>
                    <p>Importez des documents dans la barre laterale, puis posez votre premiere question.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Affichage de l'historique avec st.chat_message
        for i, turn in enumerate(st.session_state.dialogue):
            # Message utilisateur
            with st.chat_message("user"):
                badge_cls = "filter-on" if turn["filtre"] else "filter-off"
                badge_txt = "Filtre ON" if turn["filtre"] else "Filtre OFF"
                badge_dot = "●" if turn["filtre"] else "○"
                st.markdown(
                    f'{turn["question"]} '
                    f'<span class="filter-badge {badge_cls}">{badge_dot} {badge_txt}</span>',
                    unsafe_allow_html=True,
                )

            # Reponse du chatbot
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(turn["reponse"]["result"])

                # Sources dans un expander
                sources = turn["reponse"].get("source_documents", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for doc in sources:
                            src_name = doc.metadata.get("source", "Inconnu")
                            src_page = doc.metadata.get("page", "—")
                            st.markdown(
                                f"""<div class="source-card">
                                    <div class="source-meta">📄 {os.path.basename(str(src_name))} &nbsp;·&nbsp; Page {src_page}</div>
                                    <div class="source-text">{doc.page_content[:500]}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

        # Zone de saisie (st.chat_input reste ancre en bas de page)
        if query := st.chat_input("Posez votre question..."):
            mode = "filtre" if st.session_state.filtre_actif else "default"

            # Afficher immediatement le message utilisateur
            with st.chat_message("user"):
                badge_cls = "filter-on" if st.session_state.filtre_actif else "filter-off"
                badge_txt = "Filtre ON" if st.session_state.filtre_actif else "Filtre OFF"
                badge_dot = "●" if st.session_state.filtre_actif else "○"
                st.markdown(
                    f'{query} '
                    f'<span class="filter-badge {badge_cls}">{badge_dot} {badge_txt}</span>',
                    unsafe_allow_html=True,
                )

            # Initialiser RAG si necessaire
            top_k = st.session_state.top_k
            if not st.session_state.rag_initialized or st.session_state.rag_mode != mode or st.session_state.get("rag_top_k") != top_k or self.app.rag is None:
                with st.status("Initialisation du RAG...", expanded=True) as status:
                    st.write("Chargement des modeles...")
                    self.app.lancement_RAG(
                        "llama3.2:3b", "mistral:7b-instruct", mode_retriever=mode, top_k=top_k
                    )
                    status.update(label="RAG pret !", state="complete")
                st.session_state.rag_initialized = True
                st.session_state.rag_mode = mode
                st.session_state.rag_top_k = top_k

            # Obtenir la reponse
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Reflexion en cours..."):
                    response = self.app.question_reponse_rag(query)

                if response is not None:
                    st.markdown(response["result"])

                    sources = response.get("source_documents", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for doc in sources:
                                src_name = doc.metadata.get("source", "Inconnu")
                                src_page = doc.metadata.get("page", "—")
                                st.markdown(
                                    f"""<div class="source-card">
                                        <div class="source-meta">📄 {os.path.basename(str(src_name))} &nbsp;·&nbsp; Page {src_page}</div>
                                        <div class="source-text">{doc.page_content[:500]}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.dialogue.append(
                        {
                            "question": query,
                            "reponse": response,
                            "filtre": st.session_state.filtre_actif,
                            "mode": mode,
                        }
                    )

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    def sidebar(self):
        fichiers = gf.find_all_files(CHEMIN_FICHIER_RAG)

        with st.sidebar:
            # Titre sidebar
            st.markdown(
                '<div class="sidebar-title">'
                '<span class="sidebar-title-icon">📁</span>'
                'Documents'
                '</div>',
                unsafe_allow_html=True,
            )

            # Stats
            nb_fichiers = len(fichiers)
            st.markdown(
                f"""<div class="stat-row">
                    <div class="stat-pill">
                        <span class="stat-num">{nb_fichiers}</span>
                        <span class="stat-label">document{"s" if nb_fichiers != 1 else ""}</span>
                    </div>
                    <div class="stat-pill">
                        <span class="stat-num">{len(extension)}</span>
                        <span class="stat-label">formats</span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Upload
            st.markdown('<p class="section-label">Importer</p>', unsafe_allow_html=True)
            self.upload_fichier("A")

            st.divider()

            # Liste des documents existants
            st.markdown('<p class="section-label">Bibliotheque</p>', unsafe_allow_html=True)

            if fichiers:
                for i, fichier in enumerate(fichiers):
                    ext = os.path.splitext(fichier)[1].lower()
                    icon_map = {
                        ".pdf": "📕",
                        ".txt": "📝",
                        ".md": "📘",
                        ".py": "🐍",
                        ".json": "📊",
                        ".docx": "📄",
                        ".csv": "📈",
                    }
                    icon = icon_map.get(ext, "📄")

                    col_file, col_del = st.columns([0.78, 0.22], gap="small")
                    with col_file:
                        st.markdown(
                            f'<div class="file-card">'
                            f'<span class="file-icon">{icon}</span>'
                            f'<span class="file-name">{fichier}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with col_del:
                        if st.button("✕", key=f"del_{i}", help=f"Supprimer {fichier}", use_container_width=True):
                            self.app.delete_files(fichier)
                            st.session_state.rag_initialized = False
                            st.rerun()
            else:
                st.info("Aucun document encore importe.", icon="📂")

            # Parametres RAG
            st.divider()
            st.markdown('<p class="section-label">Parametres RAG</p>', unsafe_allow_html=True)
            st.session_state.top_k = st.slider(
                "Nombre de sources (k)",
                min_value=3,
                max_value=10,
                value=st.session_state.top_k,
                help="Nombre de chunks recuperes pour chaque question"
            )

            # Formats supportes
            st.divider()
            with st.expander("ℹ️ Formats supportes"):
                cols = st.columns(2)
                for idx, ext_name in enumerate(extension):
                    with cols[idx % 2]:
                        st.markdown(f"`{ext_name}`")

    # ─── Template principal ──────────────────────────────────────────────────
    def template(self):
        self.titre()
        self.discuter()
        self.sidebar()


app = App()
app.template()
