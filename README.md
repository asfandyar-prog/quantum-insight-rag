# Quantum Insight RAG

Quantum Insight RAG is a production-deployed Retrieval-Augmented Generation (RAG) system designed to support an Introduction to Quantum Computing course. Unlike generic AI chatbots, this system generates answers strictly grounded in real quantum computing research papers and course materials.

It combines:
- Semantic document retrieval
- Dense vector embeddings
- Persistent vector storage
- LLM-based contextual generation
- Cloud deployment

The result is a document-grounded AI assistant tailored for quantum education.

## Motivation
As the instructor of an Introduction to Quantum Computing course, I aimed to:
- Reduce hallucinated explanations
- Encourage research-based learning
- Provide students with an AI assistant grounded in academic materials
- Bridge AI systems with quantum education

This project explores the intersection of AI Systems × Quantum Computing × Intelligent Educational Tools.

## System Architecture
1. **Document Ingestion**
   - Supports: .pdf and .txt
   - Uses PyMuPDFLoader and TextLoader
   - Automatically detects file types
   - Adds metadata (source type, content length, index)

2. **Adaptive Chunking**
   - PDF files → Larger chunk size
   - Text files → Smaller chunk size
   - Custom recursive splitting strategy
   - Overlap handling for contextual continuity

3. **Embeddings**
   - Model: all-MiniLM-L6-v2
   - Normalized embeddings
   - Batch processing
   - Efficient inference pipeline

4. **Vector Database**
   - ChromaDB persistent storage
   - Custom collection (quantum_collection)
   - Metadata-enhanced indexing
   - Cosine similarity retrieval

5. **Retriever**
   - Top-k semantic search
   - Distance-to-similarity conversion
   - Threshold filtering
   - Ranked contextual retrieval

6. **LLM Layer**
   - Model: LLaMA 3.3 (70B) via Groq
   - Strict context-based answering
   - Hallucination control
   - Concise, technically precise outputs

7. **Deployment**
   - Hosted on Streamlit Cloud
   - Python runtime configured
   - Production-ready requirements
   - Secret management via Streamlit Secrets

## Tech Stack
- Python 3.10
- Streamlit
- LangChain
- ChromaDB
- Sentence-Transformers
- Groq (LLaMA 3.3-70B)
- PyMuPDF
- NumPy

## Example Use Cases
The system can answer questions such as:
- What is superposition in quantum mechanics?
- Explain Pauli-X gate using matrix representation.
- What is the Bernstein–Vazirani algorithm?
- How does measurement collapse the quantum state?
- Explain entanglement using tensor product notation.

All answers are generated only from ingested documents. If information is not present in the dataset, the system responds: "I don't know based on the provided documents."

## Project Structure

quantum-insight-rag/
│
├── app.py                  # Streamlit frontend
├── main.py                 # RAG pipeline backend
├── data/                   # PDFs and text documents
├── requirements.txt        # Deployment dependencies
├── runtime.txt             # Python version control
└── README.md

## Local Setup
1. Clone Repository
git clone https://github.com/asfandyar-prog/quantum-insight-rag.git
cd quantum-insight-rag
text2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
text3. Install Dependencies
pip install -r requirements.txt
text4. Add Environment Variables  
Create .env file:
GROQ_API_KEY=your_groq_api_key
text5. Run Application
streamlit run app.py
text## Deployment
The application is deployed and live on Streamlit Cloud with:
- Python 3.10 runtime
- Secrets management
- Production embedding indexing
- Persistent vector storage

**Live Demo**: [Quantum Insight RAG](https://quantum-insight-rag-uevzvmkpjuczl7lvz5xmw.streamlit.app)

## Engineering Challenges Solved
- Python 3.13 incompatibility with Torch
- Dependency conflicts in cloud deployment
- Secret scanning and push protection
- Vector store persistence issues
- Dynamic chunking optimization
- Clean separation of frontend and backend logic

## Future Improvements
- Citation display per answer
- Source filtering (PDF vs Notes)
- Hybrid search (keyword + vector)
- Multi-model support
- Evaluation harness (Faithfulness / Relevancy scoring)
- Quantum algorithm visualization integration

## About the Author
Asfand Yar  
BSc Computer Science (Minor in Physics)  
Instructor – Introduction to Quantum Computing  
AI Systems & RAG Engineer  

Interests:  
- AI Engineering  
- Retrieval-Augmented Generation  
- ML Infrastructure  
- Quantum Computing  
- AI in Education  

## License
This project is for educational and research purposes. Licensed under the MIT License.
