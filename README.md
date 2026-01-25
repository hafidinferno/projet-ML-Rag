# Agent Contextuel d'Assistance Bancaire (Fraude)

Agent RAG intelligent pour guider les clients victimes de fraude bancaire, basé sur la documentation interne de la banque.

## 🎯 Fonctionnalités

- **RAG Hybride**: Recherche sémantique (ChromaDB) + lexicale (BM25)
- **Anti-Hallucination**: L'agent ne répond qu'avec les informations des documents
- **Citations auditables**: Chaque information est tracée (chunk_id, score, source)
- **Protection anti-injection**: Filtrage des tentatives de manipulation
- **Intégration n8n**: Workflow prêt à l'emploi

## 📁 Structure du Projet

```
projet/
├── app/
│   ├── main.py              # API FastAPI
│   ├── config.py            # Configuration Pydantic
│   ├── models/              # Modèles request/response
│   ├── services/            # Logique métier
│   │   ├── ingestion.py     # Extraction PDF/MD
│   │   ├── embeddings.py    # Sentence-transformers
│   │   ├── retrieval.py     # Recherche hybride
│   │   └── agent.py         # Agent Mistral
│   ├── prompts/             # Prompts système/utilisateur
│   └── utils/               # Validators, logging
├── data/docs/               # Documents bancaires (PDF/MD)
├── n8n/                     # Workflow n8n
├── tests/                   # Tests unitaires
├── .env                     # Configuration
├── requirements.txt         # Dépendances
└── docker-compose.yml       # Déploiement
```

## 🚀 Installation & Lancement

### Prérequis
- Python 3.11 (testé sur Windows 10/11)
- Ollama avec Mistral (`ollama pull mistral`)

### ⚠️ Compatibilité des dépendances (IMPORTANT)

Ce projet utilise des versions **strictement pinnées** pour éviter les conflits connus :

| Package | Version | Raison |
|---------|---------|--------|
| **numpy** | 1.26.4 | NumPy 2.x casse ChromaDB (`np.float_` supprimé) |
| **chromadb** | 0.4.22 | Stable avec numpy 1.26 |
| **sentence-transformers** | 2.2.2 | Compatible avec huggingface_hub 0.21 |
| **huggingface_hub** | 0.21.4 | Dernière version avec `cached_download` |
| **torch** | 2.1.2 | Compatible numpy 1.26, CPU-only par défaut |

> **⚡ Premier lancement** : Le modèle d'embeddings (~100MB) sera téléchargé automatiquement. Ensuite, tout fonctionne offline.

### Installation locale (CLEAN INSTALL recommandé)

```bash
# 1. Créer un environnement propre
cd e:\pml2\projet
python -m venv .venv
.venv\Scripts\activate

# 2. Désinstaller les versions conflictuelles (si existantes)
pip uninstall -y numpy chromadb sentence-transformers huggingface_hub torch

# 3. Installer avec contraintes
pip install -r requirements.txt -c constraints.txt

# 4. Vérifier Ollama (dans un autre terminal)
ollama run mistral

# 5. Lancer l'API
python -m uvicorn app.main:app --reload --port 8000

# 6. Indexer les documents
curl -X POST http://localhost:8000/ingest
```

### Installation simple (si environnement vierge)

```bash
cd e:\pml2\projet
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Avec Docker

```bash
# Lancer l'API (Ollama doit tourner sur l'hôte)
docker-compose up fraud-agent-api

# Ou avec n8n inclus
docker-compose up
```

## 📡 Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Statut de l'API, Ollama, et index |
| `/ingest` | POST | Réindexer tous les documents |
| `/chat` | POST | Conversation avec l'agent |
| `/logs` | GET | Consulter les logs récents |
| `/documents` | GET | Lister les documents indexés |

### Exemple de requête `/chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Oui, c est une fraude. Que dois-je faire?",
    "fraud_confirmed": true,
    "transaction_context": {
      "amount": 149.99,
      "currency": "EUR",
      "merchant": "AMAZON EU",
      "channel": "online",
      "date": "2026-01-20",
      "country": "LU"
    }
  }'
```

### Exemple de réponse

```json
{
  "success": true,
  "agent_response": {
    "customer_message": "Je comprends que vous êtes victime d'une fraude...",
    "actions": [
      "1. Faites opposition à votre carte immédiatement",
      "2. Contestez l'opération via votre espace bancaire"
    ],
    "citations": [
      {
        "chunk_id": "abc123",
        "doc_id": "opposition_carte_bancaire",
        "title": "Opposition Carte Bancaire",
        "page_or_section": "Procédure",
        "excerpt": "Pour faire opposition...",
        "score": 0.87,
        "source_path": "/app/data/docs/opposition_carte_bancaire.md"
      }
    ],
    "missing_info_questions": [],
    "risk_flags": [],
    "info_not_found": false
  },
  "session_id": "a1b2c3d4",
  "processing_time_ms": 2340
}
```

## 🔗 Intégration n8n

1. Ouvrez n8n (http://localhost:5678 si docker-compose)
2. Importez `n8n/workflow_fraud_agent.json`
3. Configurez l'URL de l'API si différent de `http://localhost:8000`
4. Activez le workflow

**Webhook URL**: `POST http://localhost:5678/webhook/fraud-webhook`

**Payload attendu**:
```json
{
  "user_message": "Oui c'est une fraude",
  "fraud_confirmed": true,
  "transaction_context": { ... }
}
```

## 📚 Ajouter des Documents

Placez vos documents dans `data/docs/`:
- **PDF**: Extraction automatique par page
- **Markdown**: Extraction par section (headers)

Puis réindexez:
```bash
curl -X POST http://localhost:8000/ingest -d '{"force_reindex": true}'
```

## ⚙️ Configuration (.env)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `OLLAMA_BASE_URL` | http://localhost:11434 | URL Ollama |
| `OLLAMA_MODEL` | mistral | Modèle LLM |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Modèle embeddings |
| `CHUNK_SIZE` | 500 | Taille des chunks |
| `TOP_K_SEMANTIC` | 5 | Résultats recherche sémantique |
| `TOP_K_BM25` | 3 | Résultats recherche BM25 |
| `HYBRID_SEMANTIC_WEIGHT` | 0.7 | Poids sémantique (0-1) |

## 🧪 Tests

```bash
cd e:\pml2\projet
python -m pytest tests/ -v
```

## 🔒 Sécurité

- **Anti-injection**: Patterns détectés et marqués `untrusted`
- **Confidentialité**: Pas de PAN, CVV, PIN, mots de passe
- **Citations**: Chaque info tracée à sa source

## 📝 Exemples de Scénarios

### Scénario 1: Fraude CB en ligne

**Entrée**:
```json
{
  "user_message": "Oui, je confirme que c'est une fraude. Je n'ai jamais commandé sur ce site.",
  "fraud_confirmed": true,
  "transaction_context": {
    "amount": 299.99,
    "currency": "EUR",
    "merchant": "UNKNOWN-SHOP.COM",
    "channel": "online",
    "date": "2026-01-22"
  }
}
```

**Sortie attendue**: Instructions d'opposition + contestation + citations des documents pertinents.

### Scénario 2: Fraude virement

**Entrée**:
```json
{
  "user_message": "Oui c'est une fraude, on m'a arnaqué",
  "fraud_confirmed": true,
  "transaction_context": {
    "amount": 1500.00,
    "currency": "EUR",
    "merchant": "Virement vers IBAN inconnu",
    "channel": "virement",
    "date": "2026-01-21"
  }
}
```

**Sortie attendue**: Procédure recall virement + dépôt plainte + alerte compte compromis.

---

Développé avec ❤️ pour la sécurité bancaire.
