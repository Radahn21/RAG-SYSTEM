"""
PDF and text file extraction module.

Provides functionality to extract text content from PDF files
(using PyPDF2) and plain text files, with page number tracking.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

logger = logging.getLogger(__name__)


@dataclass
class ExtractedText:
    """Result of text extraction from a file."""
    text: str                              # Full extracted text
    source_file: str                       # Original filename
    file_path: Path                        # Path to the file
    file_type: str                         # File type (pdf, txt, md)
    page_texts: List[Tuple[int, str]] = field(default_factory=list)  # [(page_num, text), ...]
    total_pages: int = 0                   # Total number of pages (for PDFs)
    extraction_errors: List[str] = field(default_factory=list)  # Any errors during extraction


def extract_pdf_text(file_path: Path) -> ExtractedText:
    """
    Extract text from a PDF file.
    
    Uses PyPDF2 to extract text from each page, tracking page numbers
    for metadata purposes.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        ExtractedText: Extracted text with page information
    """
    logger.debug(f"Extracting text from PDF: {file_path}")
    
    result = ExtractedText(
        text="",
        source_file=file_path.name,
        file_path=file_path,
        file_type="pdf"
    )
    
    try:
        reader = PdfReader(file_path)
        result.total_pages = len(reader.pages)
        
        all_text_parts = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                
                if page_text:
                    result.page_texts.append((page_num, page_text))
                    all_text_parts.append(page_text)
                else:
                    logger.debug(f"  Page {page_num}: No text extracted (possibly image-based)")
                    
            except Exception as e:
                error_msg = f"Error extracting page {page_num}: {str(e)}"
                logger.warning(error_msg)
                result.extraction_errors.append(error_msg)
        
        # Join all page texts with double newline
        result.text = "\n\n".join(all_text_parts)
        
        logger.info(
            f"Extracted {len(result.text)} chars from {result.total_pages} pages: {file_path.name}"
        )
        
    except PdfReadError as e:
        error_msg = f"Failed to read PDF {file_path}: {e}"
        logger.error(error_msg)
        result.extraction_errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error reading {file_path}: {e}"
        logger.error(error_msg)
        result.extraction_errors.append(error_msg)
    
    return result


def extract_text_file(file_path: Path) -> ExtractedText:
    """
    Extract text from a plain text or markdown file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        ExtractedText: Extracted text
    """
    logger.debug(f"Reading text file: {file_path}")
    
    file_type = "md" if file_path.suffix.lower() in {".md", ".markdown"} else "txt"
    
    result = ExtractedText(
        text="",
        source_file=file_path.name,
        file_path=file_path,
        file_type=file_type,
        total_pages=1  # Text files are treated as single page
    )
    
    # Try different encodings
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                result.text = f.read().strip()
                result.page_texts = [(1, result.text)]
                
            logger.info(
                f"Read {len(result.text)} chars from {file_path.name} (encoding: {encoding})"
            )
            break
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            error_msg = f"Error reading {file_path}: {e}"
            logger.error(error_msg)
            result.extraction_errors.append(error_msg)
            break
    else:
        error_msg = f"Could not decode {file_path} with any supported encoding"
        logger.error(error_msg)
        result.extraction_errors.append(error_msg)
    
    return result


def extract_text(file_path: Path) -> ExtractedText:
    """
    Extract text from a file based on its extension.
    
    Supports:
    - PDF files (.pdf)
    - Text files (.txt)
    - Markdown files (.md, .markdown)
    
    Args:
        file_path: Path to the file
        
    Returns:
        ExtractedText: Extracted text with metadata
        
    Raises:
        ValueError: If file type is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext in {".txt", ".md", ".markdown"}:
        return extract_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_page_range_for_position(
    extracted: ExtractedText,
    start_char: int,
    end_char: int
) -> str:
    """
    Determine which pages a text span covers.
    
    Given character positions in the full text, determine which
    pages those characters came from.
    
    Args:
        extracted: The ExtractedText object
        start_char: Start character position in full text
        end_char: End character position in full text
        
    Returns:
        str: Page range string (e.g., "p3", "p3-p5", or "" for non-PDFs)
    """
    if extracted.file_type != "pdf" or not extracted.page_texts:
        return ""
    
    # Build a mapping of character positions to pages
    current_pos = 0
    pages_in_span = set()
    
    for page_num, page_text in extracted.page_texts:
        page_start = current_pos
        page_end = current_pos + len(page_text)
        
        # Check if this page overlaps with the span
        if page_start < end_char and page_end > start_char:
            pages_in_span.add(page_num)
        
        # Move position (account for the \n\n separator between pages)
        current_pos = page_end + 2
        
        # Stop if we've passed the end of the span
        if page_start > end_char:
            break
    
    if not pages_in_span:
        return ""
    
    pages = sorted(pages_in_span)
    
    if len(pages) == 1:
        return f"p{pages[0]}"
    else:
        return f"p{pages[0]}-p{pages[-1]}"


if __name__ == "__main__":
    # Test extraction
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: py -m src.pdf_text <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    print(f"Extracting text from: {file_path}")
    print("=" * 50)
    
    result = extract_text(file_path)
    
    print(f"File type: {result.file_type}")
    print(f"Total pages: {result.total_pages}")
    print(f"Total characters: {len(result.text)}")
    print(f"Errors: {len(result.extraction_errors)}")
    
    if result.extraction_errors:
        print("\nExtraction errors:")
        for err in result.extraction_errors:
            print(f"  - {err}")
    
    print("\nFirst 500 characters of extracted text:")
    print("-" * 50)
    print(result.text[:500])
    print("-" * 50)
