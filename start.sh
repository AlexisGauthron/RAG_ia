#!/bin/bash
# ============================================================
# Script de demarrage du projet RAG-IA (Linux/macOS)
# ============================================================

set -e  # Arreter le script en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Chemin du script (pour etre sur d'etre dans le bon dossier)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}        RAG-IA - Demarrage du projet        ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Verification de Python
echo -e "${YELLOW}[1/4] Verification de Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}  Python trouve: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}  Erreur: Python 3 n'est pas installe.${NC}"
    echo -e "${RED}  Veuillez installer Python 3.12 ou superieur.${NC}"
    exit 1
fi

# Verification de Poetry
echo -e "${YELLOW}[2/4] Verification de Poetry...${NC}"
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version 2>&1 | cut -d' ' -f3)
    echo -e "${GREEN}  Poetry trouve: $POETRY_VERSION${NC}"
else
    echo -e "${YELLOW}  Poetry non trouve. Installation...${NC}"
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}  Poetry installe avec succes.${NC}"
fi

# Installation des dependances
echo -e "${YELLOW}[3/4] Installation des dependances...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}  Creation de l'environnement virtuel...${NC}"
    poetry install
    echo -e "${GREEN}  Dependances installees.${NC}"
else
    echo -e "${GREEN}  Environnement virtuel existant detecte.${NC}"
    echo -e "${YELLOW}  Mise a jour des dependances...${NC}"
    poetry install --no-interaction
fi

# Verification d'Ollama
echo -e "${YELLOW}[4/4] Verification d'Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}  Ollama trouve.${NC}"

    # Verifier si Ollama est en cours d'execution
    if ! pgrep -x "ollama" > /dev/null; then
        echo -e "${YELLOW}  Demarrage d'Ollama en arriere-plan...${NC}"
        ollama serve &> /dev/null &
        sleep 2
    fi

    # Verifier les modeles installes
    echo -e "${YELLOW}  Verification des modeles...${NC}"
    if ! ollama list | grep -q "llama3.2"; then
        echo -e "${YELLOW}  Telechargement du modele llama3.2:3b...${NC}"
        ollama pull llama3.2:3b
    fi
    echo -e "${GREEN}  Modeles prets.${NC}"
else
    echo -e "${YELLOW}  Ollama non trouve.${NC}"
    echo -e "${YELLOW}  Pour utiliser les LLM locaux, installez Ollama: https://ollama.ai${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Demarrage de l'application Streamlit...${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Lancement de l'application Streamlit avec ouverture automatique du navigateur
poetry run streamlit run src/front/app.py --server.headless false --browser.gatherUsageStats false
