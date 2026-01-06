import pdfplumber
from typing import Optional
from pathlib import Path


class TextExtractor:
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                full_text = "\n".join(text_parts)

                if not full_text.strip():
                    raise ValueError("PDF contains no extractable text")

                return full_text

        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                raise ValueError("TXT file is empty")

            return text

        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return text
            except Exception as e:
                raise ValueError(f"Failed to read TXT file: {str(e)}")

        except Exception as e:
            raise ValueError(f"Failed to extract text from TXT: {str(e)}")

    @classmethod
    def extract_text(cls, file_path: str, file_extension: str) -> str:
        ext = file_extension.lower().lstrip('.')

        extractors = {
            'pdf': cls.extract_from_pdf,
            'txt': cls.extract_from_txt
        }

        extractor = extractors.get(ext)
        if not extractor:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(extractors.keys())}")

        return extractor(file_path)
