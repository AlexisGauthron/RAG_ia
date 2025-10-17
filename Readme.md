# Projet RAG (Retrivial Augmented Génération)


#### Consigne 

Mini RAG : concevez un petit chatbot basé sur un modèle RAG. Vous utiliserez un jeu de documents simple, mono-source (PDF, site web, wiki personnel...) pour construire un pipeline minimal (embedding + vector search + modèle). Le but est d’afficher une réponse en langage naturel à une question posée. Vous pouvez utiliser LangChain ou tout autre outil équivalent.


### J'utilise comme environnement virtuel *Poetry*

#### 1 / Téléchargé *poetry*

```bash
    pip install poetry
```

#### 2 / Installer les dépendances via *poetry*

```bash
    poetry install
```


<br><br>

### Utilisation des models via Ollama

#### 1 / Installer Ollama sur votre ordinateur

Commande windows
```bash
    winget install --id=Ollama.Ollama -e
```

#### 2 / Vérifier votre installation 

```bash
    ollama --version
    ollama serve
```

#### 3 / Télécharger un model 

```bash
    ollama pull llama3:8b
```

#### Models possible 


```bash
    # Liste des modèles à télécharger
    MODELS=(
    "mistral:7b-instruct"
    "llama3.2:1b"
    "llama3.2:3b"
    # "llama3.1:8b-instruct"
    # "llama2:7b-chat"
    # "mixtral:8x7b"
    # "qwen2.5:7b-instruct"
    # "gemma2:2b-instruct"
    # "phi3:mini"
)
```

<br><br>

### Utilisation Streamlit pour l'interface 

```bash
    streamlit run src/front/app.py
```

<br><br>

## Fonctionnalité :

- Chatbot RAG se basant sur plusieurs fichiers
- Utilisation LLM pour trouver les filtres correspondant à la query 
- Gestionnaire de fichier et database via ChromaDB 
    - Possibilité de supprimer, ajouter des fichiers 
    - Gestion des fichiers doublons

- Utilisation CLI (beaucoup de message [DEBUG])
- Utilisation Site Web via Streamlit


<br><br>

## Organisation Fonction 

#### ```embedding.py``` 

- load_text_file : load les datas présentes dans le dossier data

- chunk_text : Séparer en chunks les textes extrait et les regroupes dans un tableau

- class VectorIndex : 

# En cours 