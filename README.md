# Rag-soc

## 🚀 Lancer le projet RAG-Soc

### 1. Cloner le dépôt
```bash
git clone https://github.com/dicar-2001/Rag-soc.git
cd Rag-soc
```

### 2. Installer les dépendances
```bash
pip install -r requirement.txt
```

### 3. Configurer la clé API Groq
- Crée un fichier `.env` ou `api.env` à la racine du projet avec :
  ```
  GROQ_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
- **Ne partage jamais ta clé API sur GitHub !**

### 4. Préparer les documents
- Place tes fichiers `.pdf`, `.txt` ou `.docx` dans le dossier `docs/`.

### 5. Lancer OpenSearch avec Docker
```bash
docker-compose -f docker-compose-opensearch.yml up -d
```

### 6. Lancer l’application Chainlit
```bash
chainlit run app.py
```
- Ouvre ensuite [http://localhost:8000](http://localhost:8000) dans ton navigateur.

---

**Résumé des commandes principales** :
```bash
git clone <repo>
cd <repo>
pip install -r requirement.txt
# Ajouter la clé API dans .env ou api.env
docker-compose -f docker-compose-opensearch.yml up -d
chainlit run app.py
```
