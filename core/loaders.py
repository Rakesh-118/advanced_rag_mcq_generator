from typing import Union
from pypdf import PdfReader
from docx import Document
import io


class DocumentLoaderError(Exception):
    """Custom exception for document loading failures."""
    pass


def load_pdf(file: Union[io.BytesIO, str]) -> str:
    """
    Extract text from a PDF file.

    Args:
        file: Uploaded file object or file path.

    Returns:
        Extracted text as string.

    Raises:
        DocumentLoaderError: If extraction fails.
    """
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if not text.strip():
            raise DocumentLoaderError("PDF contains no extractable text.")

        return text.strip()

    except Exception as e:
        raise DocumentLoaderError(f"Failed to load PDF: {str(e)}")


def load_docx(file: Union[io.BytesIO, str]) -> str:
    """
    Extract text from a DOCX file.

    Args:
        file: Uploaded file object or file path.

    Returns:
        Extracted text as string.

    Raises:
        DocumentLoaderError: If extraction fails.
    """
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])

        if not text.strip():
            raise DocumentLoaderError("DOCX contains no readable text.")

        return text.strip()

    except Exception as e:
        raise DocumentLoaderError(f"Failed to load DOCX: {str(e)}")


def validate_text_input(text: str) -> str:
    """
    Validate raw text input.

    Raises:
        ValueError: If input is empty.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    return text.strip()
