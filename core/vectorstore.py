import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


def create_vector_store(text: str) -> FAISS:
    """
    Create FAISS vector store from input text.

    Args:
        text: Cleaned input text.

    Returns:
        FAISS vector store instance.
    """
    try:
        # Step 1: Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_text(text)

        if not chunks:
            raise VectorStoreError("Text splitting resulted in empty chunks.")

        # Step 2: Convert chunks into Documents
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Step 3: Create embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )


        # Step 4: Build FAISS index
        vector_store = FAISS.from_documents(documents, embeddings)

        return vector_store

    except Exception as e:
        raise VectorStoreError(f"Failed to create vector store: {str(e)}")


def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = 5) -> List[str]:
    """
    Retrieve top-k relevant chunks from vector store.

    Args:
        vector_store: FAISS vector store
        query: Retrieval query
        k: Number of chunks to retrieve

    Returns:
        List of relevant text chunks
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        documents = retriever.invoke(query)

        return [doc.page_content for doc in documents]

    except Exception as e:
        raise VectorStoreError(f"Failed to retrieve relevant chunks: {str(e)}")
