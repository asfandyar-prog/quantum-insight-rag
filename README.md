<div align="center">

# Quantum Insight RAG

**A production-deployed Retrieval-Augmented Generation system for quantum computing education**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Deployed](https://img.shields.io/badge/Status-Production%20Deployed-22c55e?style=flat)
![RAG](https://img.shields.io/badge/Architecture-RAG-6366f1?style=flat)
![LangChain](https://img.shields.io/badge/Framework-LangChain-f97316?style=flat)
![License](https://img.shields.io/badge/License-MIT-0ea5e9?style=flat)

*Strictly grounded in real quantum computing research papers and course materials — not a generic chatbot.*

</div>

----

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Example Queries](#-example-queries)
- [Design Philosophy](#-design-philosophy)

-----

## 🧠 Overview

**Quantum Insight RAG** is a document-grounded AI assistant purpose-built for an *Introduction to Quantum Computing* course. It answers questions strictly from retrieved content — research papers and course materials — never from the model’s parametric memory alone.

|Problem                              |Solution                                              |
|-------------------------------------|------------------------------------------------------|
|LLMs hallucinate quantum concepts    |Answers grounded in retrieved source documents        |
|Generic chatbots lack course context |Embeddings built from actual course materials         |
|Students get low-quality explanations|Semantic retrieval surfaces the most relevant passages|
|AI-generated content hard to verify  |Every answer traceable back to a source chunk         |


> *Intersection of AI Systems × Quantum Computing × Intelligent Educational Tools*

-----

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
│                                                             │
│  PDF / TXT files                                            │
│       │                                                     │
│       ▼                                                     │
│  PyMuPDFLoader / TextLoader  (auto file-type detection)     │
│       │                                                     │
│       ▼                                                     │
│  RecursiveCharacterTextSplitter  (chunk + overlap)          │
│       │                                                     │
│       ▼                                                     │
│  HuggingFace Embeddings  (dense vector encoding)            │
│       │                                                     │
│       ▼                                                     │
│  ChromaDB  (persistent vector store)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                        │
│                                                             │
│  User question                                              │
│       │                                                     │
│       ▼                                                     │
│  Embed query  →  Similarity search (ChromaDB)               │
│       │                                                     │
│       ▼                                                     │
│  Top-k relevant chunks retrieved                            │
│       │                                                     │
│       ▼                                                     │
│  Prompt = context chunks + user question                    │
│       │                                                     │
│       ▼                                                     │
│  LLM (Groq / OpenAI)  →  Grounded answer                   │
└─────────────────────────────────────────────────────────────┘
```

-----

## 🔄 RAG Pipeline

### 1. Document Ingestion

```python
# Auto-detects file type and loads accordingly
if file.endswith(".pdf"):
    loader = PyMuPDFLoader(file_path)
elif file.endswith(".txt"):
    loader = TextLoader(file_path)

docs = loader.load()
```

### 2. Chunking & Embedding

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 3. Persistent Vector Storage

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./data/chroma_db"
)
vectorstore.persist()
```

### 4. Retrieval & Generation

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = chain({"query": user_question})
```

-----

## 📁 Repository Structure

```
quantum-insight-rag/
│
├── app.py                # Streamlit web interface
├── main.py               # RAG pipeline core logic
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python runtime version (Render/Heroku)
├── .gitignore
│
└── data/                 # Document corpus
    ├── papers/           # Quantum computing research papers (.pdf)
    ├── notes/            # Course materials (.txt / .pdf)
    └── chroma_db/        # Persistent vector store (auto-generated)
```

-----

## 🛠 Tech Stack

|Layer               |Technology                                          |
|--------------------|----------------------------------------------------|
|**Frontend**        |Streamlit                                           |
|**RAG Framework**   |LangChain                                           |
|**Embeddings**      |HuggingFace `sentence-transformers/all-MiniLM-L6-v2`|
|**Vector Store**    |ChromaDB (persistent)                               |
|**Document Loading**|PyMuPDF · LangChain TextLoader                      |
|**LLM**             |Groq API / OpenAI                                   |
|**Deployment**      |Cloud (Render / Streamlit Cloud)                    |
|**Language**        |Python 3.11+                                        |

-----

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A Groq or OpenAI API key

### Installation

```bash
git clone https://github.com/asfandyar-prog/quantum-insight-rag.git
cd quantum-insight-rag
pip install -r requirements.txt
```

### Configuration

```bash
# Create a .env file
echo "GROQ_API_KEY=your_key_here" > .env
# or
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Ingest your documents

```bash
# Drop PDFs or TXT files into data/
python main.py --ingest
```

### Run the app

```bash
streamlit run app.py
```

-----

## 💬 Example Queries

```
"What is quantum superposition?"
→ Retrieved from: lecture_week2.pdf, nielsen_chuang_ch2.pdf

"Explain Shor's algorithm step by step."
→ Retrieved from: shors_algorithm_paper.pdf, course_notes_week6.txt

"What are the main sources of quantum decoherence?"
→ Retrieved from: decoherence_review.pdf, lab_notes.txt
```

Every answer cites which document chunks were retrieved — students can trace the source.

-----

## 🏗 Design Philosophy

**Grounding over generation.** The system is constrained to answer from retrieved context only. If the answer isn’t in the document corpus, it says so — rather than hallucinating.

**Education-first.** Built specifically for a quantum computing course, not adapted from a generic chatbot. The document corpus, chunking strategy, and prompt design are all tuned for academic content.

**Production-deployed.** Not a notebook demo. Deployed with a persistent vector store, a web interface, and a cloud runtime — ready for real student use.

-----

## 👤 Author

**Asfand Yar** · BSc Computer Science
Focus: agentic AI · self-supervised learning · vision transformers · robustness under distribution shift

-----

<div align="center">

MIT License · [asfandyar-prog](https://github.com/asfandyar-prog)

</div>
