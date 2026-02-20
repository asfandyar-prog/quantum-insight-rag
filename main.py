from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader,TextLoader


##Lets do the data ingestion.

BASE_DIR = Path(__file__).resolve().parent
DATA_ = BASE_DIR / "data"

print("Looking for files in:", DATA_)

if not DATA_.exists():
    raise FileNotFoundError(f"Data directory not found at {DATA_}")

File_Loaders={
    "*.pdf":PyMuPDFLoader,
    "*.txt":TextLoader
}

all_docs=[]
for pattern,loader_cls in File_Loaders.items():
    loader=DirectoryLoader(
        str(DATA_)
    ,
    glob=f"**/{pattern}",
    loader_cls=loader_cls,
    show_progress=True
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"]=pattern.replace("*.","")
    
    all_docs.extend(docs)

docs=all_docs



# Make chunks
def make_chunks(documents, chunk_overlap=200):
    all_chunks = []
    for doc in documents:
        if doc.metadata.get("source") == "pdf":
            chunk_size = 1500
        else:
            chunk_size = 600

        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ',', ' ', ''],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print("Documents split into chunks")
    return all_chunks

split = make_chunks(docs)



# Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict, Any

class Embeddings:
    """handle the documents and make the embeddings of document's chunks"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Load the model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"model is successfully loaded {self.model_name}")
        except Exception as e:
            print(f"error while loading the model {e}")
            raise

    def get_embeddings(self, texts: List[str]):
        """Takes the list of chunks and make the embeddings for each chunk."""
        if not self.model:
            raise RuntimeError("Model not loaded.")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True,normalize_embeddings=True)
            return embeddings
        except Exception as e:
            print(f"Error occured while processing the embeddings of chunks {e}")
            raise



texts = [d.page_content for d in split]
embeddings_model = Embeddings()
chunk_embeddings = embeddings_model.get_embeddings(texts)



# Vector Store
class Vectors:
    """Handle the embeddings and store it in the vector base"""
    def __init__(self, collection_name="quantum_collection", persistent_dir="./quantum_vector_store"):
        self.collection_name = collection_name
        self.persistent_dir = persistent_dir
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize the vector store"""
        os.makedirs(self.persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persistent_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF documents embeddings for RAG"}
        )
        print(f"Initialized VectorStore with collection: {self.collection_name}")

    def add_documents(self, documents, embeddings: np.ndarray):
        """Add documents and their embeddings to the vector base"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        embeddings_list = []
        documents_text = []
        metadatas = []

        #  fix enumerate(zip(...)) unpack
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)  #  ids not id

            metadata = dict(doc.metadata)  #  dict(...) not Dict(...)
            metadata["index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(emb.tolist())  #  each vector, not whole array

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Added {len(documents)} docs with {len(embeddings_list)} embeddings")
        except Exception as e:
            print("error occured while handling the document and embedding storing")
            raise

    def query(self, query_embeddings: np.ndarray, top_k: int = 5):
        """handle the query and make its embeddings"""
        try:
            # chroma expects list[list[float]]
            if hasattr(query_embeddings, "tolist"):
                query_embeddings = query_embeddings.tolist()

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,  #  n_results not top_k
                include=["documents", "distances", "metadatas"]  # correct keys
            )
            return results
        except Exception as e:
            print(f"error while handling the embeddings of query. {e}")
            raise

vectorstore = Vectors()

if vectorstore.collection.count() == 0:
    print("Indexing documents...")
    vectorstore.add_documents(split, chunk_embeddings)
else:
    print("Vector store already indexed.")

# Retriever

class RetrieverAugmentedGeneration:
    def __init__(self, vectorstore: Vectors, embeddings: Embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.get_embeddings([query])  # (1, dim)

        results = self.vectorstore.collection.query(
            query_embeddings=query_embedding.tolist(),  # 
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )

        retrieved_docs = []
        for i, (doc_id, distance, doc_text, metadata) in enumerate(
            zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ):
            similarity = 1.0 - distance
            print(f"Rank {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")

            if similarity >= threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "documents": doc_text,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "distance": distance,
                    "rank": i + 1,
                })

        return retrieved_docs



# LLM + Pipeline
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

retriever = RetrieverAugmentedGeneration(vectorstore, embeddings_model)


def reg_simple(query, retriever, llm, top_k=3):
    # retrieve context
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([d["documents"] for d in results]) if results else "No context found"  # ✅ documents key

    prompt = f"""
        You are a Quantum Computing assistant.

        Answer ONLY using the provided context.
        If the answer is not contained in the context, say:
        "I don't know based on the provided documents."

        Keep answers:
        - Clear
        - Technically precise
        - Concise
        - Include equations in plain text if relevant

        Context:
        {context}

        Question:
        {query}

        Answer:
    """


    response = llm.invoke(prompt) 
    return response.content
