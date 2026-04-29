# 📚 PDF Chatbot — RAG avec LangChain & LangGraph

Un chatbot pédagogique de niveau intermédiaire qui répond à tes questions à partir de **PDFs** et de **pages web**, en s'appuyant sur une architecture RAG (Retrieval-Augmented Generation) avancée orchestrée par **LangGraph**.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 📄 **Ingestion PDF** | Upload et indexation automatique de documents PDF |
| 🌐 **Ingestion URL** | Indexation de pages web complètes |
| 🗂️ **Collections** | Organisation des documents en projets séparés |
| 🔍 **Hybrid Search** | Combinaison BM25 (mots-clés) + recherche vectorielle |
| 🎯 **Reranking** | Cross-encoder pour classer les résultats par pertinence réelle |
| ✍️ **Query Rewriting** | Reformulation contextuelle des questions de suivi |
| 🔄 **Corrective RAG** | LangGraph vérifie la pertinence et relance si nécessaire |
| 💬 **Historique** | Le chatbot se souvient du contexte de la conversation |
| 📎 **Citations** | Chaque réponse indique la source (fichier + page) |
| 🎓 **Mode professeur** | Réponses naturelles, pédagogiques, avec exemples |

---

## 🏗️ Architecture

```
Question
   │
   ▼
[Rewrite Query]     ← reformule selon l'historique de conversation
   │
   ▼
[Retrieve]          ← Hybrid Search (BM25 + ChromaDB) → Cross-encoder Reranking
   │
   ▼
[Grade Docs]        ← LLM juge la pertinence des chunks récupérés
   │
   ├── pertinent ──────────────────────────────► [Generate] → Réponse + Sources
   │
   └── non pertinent (max 2 essais) ──► [Rewrite Query] ↩
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| LLM | OpenAI GPT-4o |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace, local) |
| Vector store | ChromaDB (persistant) |
| Keyword search | BM25 (`rank-bm25`) |
| Orchestration | LangGraph |
| Loaders | LangChain (`PyPDFLoader`, `WebBaseLoader`) |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/BilalAlfayoumi/Pdf_chatboot.git
cd Pdf_chatboot
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

```bash
cp .env.example .env
# Éditer .env et ajouter ta clé OpenAI
```

Ou directement depuis l'interface Streamlit (sidebar).

### 5. Lancer l'application

```bash
streamlit run app.py
```

Ouvre [http://localhost:8501](http://localhost:8501) dans ton navigateur.

---

## 📖 Utilisation

1. **Entrer ta clé API OpenAI** dans la barre latérale
2. **Créer une collection** (ex: `Cours IA`, `Rapport stage`)
3. **Uploader des PDFs** ou **ajouter des URLs**
4. **Poser tes questions** — le chatbot répond comme un professeur

---

## 📁 Structure du projet

```
Pdf_chatboot/
├── app.py                  # Interface Streamlit
├── rag/
│   ├── ingestion.py        # Chargement, découpage, embedding → ChromaDB
│   ├── retriever.py        # Hybrid Search + Reranking
│   ├── grader.py           # Évaluation de la pertinence
│   └── graph.py            # Pipeline LangGraph (Corrective RAG)
├── requirements.txt
├── .env.example
└── chroma_db/              # Base vectorielle locale (générée automatiquement)
```

---

## ⚙️ Variables d'environnement

```env
OPENAI_API_KEY=sk-...
```

> La clé peut aussi être saisie directement dans l'interface sans fichier `.env`.

---

## 📦 Dépendances principales

```
langchain / langchain-community / langchain-openai / langchain-huggingface
langchain-classic
langgraph
chromadb
streamlit
sentence-transformers
rank-bm25
pypdf
beautifulsoup4
```

---

## 🔒 Sécurité

- Ne jamais committer le fichier `.env` (il est dans `.gitignore`)
- Le fichier `.env.example` ne contient qu'un placeholder (`sk-...`)
- En cas d'exposition accidentelle d'une clé : la révoquer immédiatement sur [platform.openai.com](https://platform.openai.com/api-keys)

---

## 👤 Auteur

**AL FAYOUMI BILAL**  
Projet RAG — LangChain / LangGraph / Streamlit
