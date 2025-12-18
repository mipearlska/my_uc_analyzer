"""
Vector Store implementations for ETSI RAG system.

Provides three backends:
- ChromaVectorStore: Local, simple, good for development
- MilvusVectorStore: Local/cloud, production-ready
- MongoDBAtlasVectorStore: Cloud, integrated with MongoDB
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.data_models import DocumentChunk, SectionType, UseCaseCategory


# =============================================================================
# Embedding Model Setup
# =============================================================================

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Get the configured embedding model."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

# =============================================================================
# Base Interface
# =============================================================================

class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    All implementations must support:
    - Adding chunks with metadata
    - Similarity search with optional metadata filtering
    """
    
    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        use_case_id: Optional[str] = None,
        section_type: Optional[SectionType] = None,
        category: Optional[UseCaseCategory] = None,
    ) -> list[DocumentChunk]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            use_case_id: Filter by use case (e.g., "5.1.1")
            section_type: Filter by DESCRIPTION or REQUIREMENTS
            category: Filter by CONSUMER, BUSINESS, or OPERATOR
            
        Returns:
            List of DocumentChunk objects, sorted by relevance
        """
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        """Delete all chunks from the vector store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks in the store."""
        pass


# =============================================================================
# Helper: Convert DocumentChunk to/from LangChain Document
# =============================================================================

from langchain_core.documents import Document


def chunk_to_document(chunk: DocumentChunk) -> Document:
    """Convert DocumentChunk to LangChain Document."""
    return Document(
        page_content=chunk.content,
        metadata={
            "chunk_id": chunk.chunk_id,
            "use_case_id": chunk.use_case_id,
            "use_case_name": chunk.use_case_name,
            "section_type": chunk.section_type.value,
            "category": chunk.category.value,
            "page_start": chunk.page_start,
            "token_count": chunk.token_count,
            "chunk_index": chunk.chunk_index,
        }
    )


def document_to_chunk(doc: Document) -> DocumentChunk:
    """Convert LangChain Document back to DocumentChunk."""
    return DocumentChunk(
        chunk_id=doc.metadata["chunk_id"],
        content=doc.page_content,
        use_case_id=doc.metadata["use_case_id"],
        use_case_name=doc.metadata["use_case_name"],
        section_type=SectionType(doc.metadata["section_type"]),
        category=UseCaseCategory(doc.metadata["category"]),
        page_start=doc.metadata["page_start"],
        token_count=doc.metadata["token_count"],
        chunk_index=doc.metadata["chunk_index"],
    )

def build_metadata_filter(
    use_case_id: Optional[str] = None,
    section_type: Optional[SectionType] = None,
    category: Optional[UseCaseCategory] = None,
) -> dict:
    """Build metadata filter dict for vector store queries."""
    filters = {}
    if use_case_id:
        filters["use_case_id"] = use_case_id
    if section_type:
        filters["section_type"] = section_type.value
    if category:
        filters["category"] = category.value
    return filters

# =============================================================================
# Chroma Implementation
# =============================================================================

from langchain_chroma import Chroma


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store implementation.
    
    - Local, file-based storage
    - Simple setup, good for development
    - Supports metadata filtering
    
    Usage:
        store = ChromaVectorStore(persist_directory="./data/chroma_db")
        store.add_chunks(chunks)
        results = store.search("What are the requirements?", k=5)
    """
    BASE_DIR = Path(__file__).resolve().parent
    CHORMADB_PATH = BASE_DIR.parent/"data"/"chroma_db"

    def __init__(
        self, 
        persist_directory: str | Path = CHORMADB_PATH,
        collection_name: str = "etsi_chunks"
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self._embeddings = get_embedding_model()
        
        # Initialize Chroma
        self._store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self._embeddings,
            collection_name=self.collection_name,
        )
    
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to Chroma."""
        if not chunks:
            return
        
        documents = [chunk_to_document(chunk) for chunk in chunks]
        
        # Add with IDs to allow updates
        ids = [chunk.chunk_id for chunk in chunks]
        
        self._store.add_documents(documents=documents, ids=ids)
    
    def search(
        self,
        query: str,
        k: int = 5,
        use_case_id: Optional[str] = None,
        section_type: Optional[SectionType] = None,
        category: Optional[UseCaseCategory] = None,
    ) -> list[DocumentChunk]:
        """Search Chroma with optional metadata filtering."""
        
        # Build filter
        where_filter = None
        filter_dict = build_metadata_filter(use_case_id, section_type, category)
        
        if filter_dict:
            # Chroma uses $and for multiple conditions
            if len(filter_dict) == 1:
                key, value = list(filter_dict.items())[0]
                where_filter = {key: {"$eq": value}}
            else:
                where_filter = {
                    "$and": [
                        {key: {"$eq": value}} 
                        for key, value in filter_dict.items()
                    ]
                }
        
        # Search
        results = self._store.similarity_search(
            query=query,
            k=k,
            filter=where_filter
        )
        
        # Convert back to DocumentChunk and sort by chunk_index
        chunks = [document_to_chunk(doc) for doc in results]
        chunks.sort(key=lambda c: c.chunk_index)
        
        return chunks
    
    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # Get all IDs and delete them
        collection = self._store._collection
        if collection.count() > 0:
            all_ids = collection.get()["ids"]
            collection.delete(ids=all_ids)
    
    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._store._collection.count()
    

# =============================================================================
# Milvus Implementation
# =============================================================================

from langchain_milvus import Milvus


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus vector store implementation.
    
    - Uses Milvus Lite for local storage (no server needed)
    - Production-ready, scalable
    - Supports metadata filtering
    
    Usage:
        store = MilvusVectorStore(uri="./data/milvus.db")
        store.add_chunks(chunks)
        results = store.search("What are the requirements?", k=5)
    """

    BASE_DIR = Path(__file__).resolve().parent
    MILVUS_PATH = BASE_DIR.parent/"data"/"milvus_db"
    
    def __init__(
        self,
        uri: str | Path = MILVUS_PATH,
        collection_name: str = "etsi_chunks"
    ):
        self.uri = str(uri)
        self.collection_name = collection_name
        self._embeddings = get_embedding_model()
        self._store: Optional[Milvus] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize Milvus connection if not already done."""
        if not self._initialized and self._store is not None:
            self._initialized = True
    
    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to Milvus."""
        if not chunks:
            return
        
        documents = [chunk_to_document(chunk) for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Create or update the store
        self._store = Milvus.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=self.collection_name,
            connection_args={"uri": self.uri},
            ids=ids,
            drop_old=True,  # Replace existing collection
        )
        self._initialized = True
    
    def search(
        self,
        query: str,
        k: int = 5,
        use_case_id: Optional[str] = None,
        section_type: Optional[SectionType] = None,
        category: Optional[UseCaseCategory] = None,
    ) -> list[DocumentChunk]:
        """Search Milvus with optional metadata filtering."""
        
        if self._store is None:
            # Try to connect to existing collection
            self._store = Milvus(
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri},
            )
        
        # Build filter expression (Milvus uses string expressions)
        filter_conditions = []
        if use_case_id:
            filter_conditions.append(f'use_case_id == "{use_case_id}"')
        if section_type:
            filter_conditions.append(f'section_type == "{section_type.value}"')
        if category:
            filter_conditions.append(f'category == "{category.value}"')
        
        expr = " and ".join(filter_conditions) if filter_conditions else None
        
        # Search
        results = self._store.similarity_search(
            query=query,
            k=k,
            expr=expr
        )
        
        # Convert back to DocumentChunk and sort by chunk_index
        chunks = [document_to_chunk(doc) for doc in results]
        chunks.sort(key=lambda c: c.chunk_index)
        
        return chunks
    
    def delete_all(self) -> None:
        """Delete collection."""
        if self._store is not None:
            self._store.delete(expr="chunk_index >= 0")  # Delete all
    
    def count(self) -> int:
        """Return number of documents in collection."""
        if self._store is None:
            return 0
        try:
            # Access underlying collection
            collection = self._store.col
            if collection:
                return collection.num_entities
        except Exception:
            pass
        return 0
    