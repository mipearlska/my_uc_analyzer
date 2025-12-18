"""
ETSI Document Parser using Docling.

Extracts use case chunks from ETSI specification PDFs.
Outputs DocumentChunk objects ready for RAG.
"""

import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from src.data_models import (
    DocumentChunk,
    SectionType,
    UseCaseCategory,
    get_use_case_category,
)


# Pattern to match use case content sections: 5.x.x.1 (Description) or 5.x.x.2 (Requirements)
USE_CASE_CONTENT_PATTERN = re.compile(r'^(5\.\d+\.\d+)\.([12])\s+(.+)$')

# Pattern to extract section ID from heading
SECTION_ID_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')


class ETSIParser:
    """
    Parser for ETSI specification documents using Docling.
    
    Extracts only use case Description and Requirements sections,
    returning DocumentChunk objects ready for embedding.
    
    Usage:
        parser = ETSIParser()
        chunks = parser.parse("path/to/etsi.pdf")
        
        for chunk in chunks:
            print(f"{chunk.use_case_id}: {chunk.content[:100]}...")
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._converter = DocumentConverter()
        self._chunker = HybridChunker()
        self._use_case_names: dict[str, str] = {}  # Cache: "5.1.1" -> "AI Agents to Enable Smart Life"
        
    def _log(self, message: str):
        if self.verbose:
            print(f"[ETSIParser] {message}")
    
    def parse(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """
        Parse an ETSI PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects (only Description and Requirements sections)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self._log(f"Parsing: {pdf_path.name}")
        
        # Step 1: Convert with Docling
        self._log("Converting document with Docling...")
        result = self._converter.convert(source=str(pdf_path))
        doc = result.document
        
        # Step 2: Extract use case names from section headers
        self._log("Extracting use case names...")
        self._extract_use_case_names(doc)
        self._log(f"Found {len(self._use_case_names)} use cases")
        
        # Step 3: Create chunks using HybridChunker
        self._log("Creating chunks...")
        raw_chunks = list(self._chunker.chunk(dl_doc=doc))
        self._log(f"Docling created {len(raw_chunks)} total chunks")
        
        # Step 4: Filter and convert to DocumentChunk
        self._log("Filtering to use case content only...")
        chunks = self._process_chunks(raw_chunks)
        self._log(f"Extracted {len(chunks)} use case chunks")
        
        return chunks
    
    def _extract_use_case_names(self, doc):
        """
        Extract use case names from section headers.
        
        Looks for patterns like:
        - 5.1.1 Use Case: AI Agents to Enable Smart Life
        - 5.2.1 Use Case on AI Agent-based Customized Network...
        """
        self._use_case_names = {}
        
        for text_item in doc.texts:
            if text_item.label != "section_header":
                continue
            
            match = SECTION_ID_PATTERN.match(text_item.text)
            if not match:
                continue
            
            section_id = match.group(1)
            title = match.group(2)
            
            # Match use case level sections: 5.1.1, 5.2.1, etc. (exactly 3 levels)
            parts = section_id.split('.')
            if len(parts) == 3 and section_id.startswith('5.'):
                # Clean up the title
                use_case_name = title.strip()
                # Remove "Use Case:" or "Use Case on" prefix if present
                use_case_name = re.sub(r'^Use Case[:\s]+', '', use_case_name, flags=re.IGNORECASE)
                use_case_name = re.sub(r'^on\s+', '', use_case_name, flags=re.IGNORECASE)
                
                self._use_case_names[section_id] = use_case_name.strip()
    
    def _process_chunks(self, raw_chunks) -> list[DocumentChunk]:
        """
        Filter raw chunks to only use case content and convert to DocumentChunk.
        """
        chunks = []
        chunk_counter = 0  # Global counter for chunk_index
        
        for raw_chunk in raw_chunks:
            # Get heading from metadata
            heading = ""
            if hasattr(raw_chunk, 'meta') and raw_chunk.meta.headings:
                heading = raw_chunk.meta.headings[0]
            
            # Check if this is a use case content section (5.x.x.1 or 5.x.x.2)
            match = USE_CASE_CONTENT_PATTERN.match(heading)
            if not match:
                continue
            
            use_case_id = match.group(1)      # e.g., "5.1.1"
            section_num = match.group(2)       # "1" for Description, "2" for Requirements
            
            # Determine section type
            if section_num == "1":
                section_type = SectionType.DESCRIPTION
            else:
                section_type = SectionType.REQUIREMENTS
            
            # Get use case name
            use_case_name = self._use_case_names.get(use_case_id, "Unknown")
            
            # Get category
            try:
                category = get_use_case_category(use_case_id)
            except ValueError:
                category = UseCaseCategory.CONSUMER
            
            # Get page number
            page = 0
            if (hasattr(raw_chunk, 'meta') and 
                raw_chunk.meta.doc_items and 
                raw_chunk.meta.doc_items[0].prov):
                page = raw_chunk.meta.doc_items[0].prov[0].page_no
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                chunk_id=f"chunk_{use_case_id}_{chunk_counter:03d}",
                content=raw_chunk.text,
                use_case_id=use_case_id,
                use_case_name=use_case_name,
                section_type=section_type,
                category=category,
                page_start=page,
                token_count=len(raw_chunk.text) // 4,  # Rough estimate
                chunk_index=chunk_counter
            )
            chunks.append(chunk)
            chunk_counter += 1
        
        return chunks


def parse_etsi_document(pdf_path: str | Path) -> list[DocumentChunk]:
    """Convenience function to parse an ETSI document."""
    parser = ETSIParser()
    return parser.parse(pdf_path)