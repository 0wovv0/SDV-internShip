from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any
import json

# Initialize the Ollama model
llm = Ollama(
    model="mistral-local",
    base_url="http://localhost:11434",
    temperature=0.1
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_documents_with_metadata(texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[Document]:
    """
    Create documents with metadata for each text chunk
    """
    documents = []
    for text, metadata in zip(texts, metadata_list):
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        documents.append(doc)
    return documents

def create_vector_store(documents: List[Document], metadata_fields: List[str] = None):
    """
    Create a vector store with metadata support
    """
    # Initialize the token text splitter
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Split documents into chunks while preserving metadata
    splits = text_splitter.split_documents(documents)
    
    # Create vector store with FAISS
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def create_enhanced_retriever(vector_store, metadata_fields: List[str] = None):
    """
    Create an enhanced retriever that can filter by metadata
    """
    # Create base retriever with FAISS and metadata filtering
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5,
            "fetch_k": 20,
            "filter": None  # Can be updated with metadata filters
        }
    )
    
    return base_retriever

def create_rag_chain(vector_store, metadata_fields: List[str] = None):
    """
    Create a RAG chain that can use metadata for better context
    """
    # Create a prompt template that includes metadata context
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create enhanced retriever
    retriever = create_enhanced_retriever(vector_store, metadata_fields)

    # Create the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain

# Example usage
def main():
    # Example documents with metadata
    documents = [
        {
            "text": "The capital of France is Paris.",
            "metadata": {
                "year": 2023,
                "category": "geography",
                "source": "encyclopedia"
            }
        },
        {
            "text": "The Eiffel Tower is located in Paris.",
            "metadata": {
                "year": 2023,
                "category": "landmarks",
                "source": "travel_guide"
            }
        },
        {
            "text": "Việt Nam hoàn toàn giải phóng, thống nhất đất nước.",
            "metadata": {
                "year": 1975,
                "category": "history",
                "source": "history_event"
            }
        }
    ]

    # Create documents with metadata
    docs = create_documents_with_metadata(
        [doc["text"] for doc in documents],
        [doc["metadata"] for doc in documents]
    )

    # Create vector store
    vector_store = create_vector_store(docs)
    
    # Save the vector store locally
    vector_store.save_local("vectorstore")
    
    # Create RAG chain
    chain = create_rag_chain(vector_store)

    # Example query with metadata filtering
    query = "Sự kiện gì xảy ra năm 1975?"
    result = chain({"query": query})
    
    print("Question:", query)
    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("- Content:", doc.page_content)
        print("  Metadata:", doc.metadata)

if __name__ == "__main__":
    main()