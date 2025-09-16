"""Document preprocessing utilities."""

import re
from typing import Dict, Any, List
from dataclasses import dataclass

from src.core.interfaces import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for document preprocessing."""
    remove_extra_whitespace: bool = True
    remove_empty_lines: bool = True
    normalize_unicode: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    remove_special_chars: bool = False
    min_line_length: int = 10
    max_line_length: int = 1000
    remove_code_blocks: bool = False
    preserve_structure: bool = True


class DocumentPreprocessor:
    """Preprocesses documents for better indexing and retrieval."""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()

    async def preprocess(self, document: Document) -> Document:
        """
        Preprocess a document.
        
        Args:
            document: Document to preprocess
            
        Returns:
            Preprocessed document
        """
        try:
            content = document.content
            original_length = len(content)

            # Apply preprocessing steps
            content = self._normalize_unicode(content)
            content = self._clean_whitespace(content)
            content = self._remove_unwanted_patterns(content)
            content = self._filter_lines(content)
            content = self._preserve_structure(content)

            processed_length = len(content)
            
            logger.debug(
                f"Preprocessed document {document.source}: "
                f"{original_length} -> {processed_length} chars"
            )

            # Create new document with preprocessed content
            return Document(
                content=content,
                source=document.source,
                chunk_id=document.chunk_id,
                metadata={
                    **document.metadata,
                    "preprocessed": True,
                    "original_length": original_length,
                    "processed_length": processed_length,
                },
                embedding=document.embedding,
                similarity_score=document.similarity_score,
            )

        except Exception as e:
            logger.error(f"Error preprocessing document {document.source}: {e}")
            # Return original document if preprocessing fails
            return document

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        if not self.config.normalize_unicode:
            return text

        import unicodedata
        
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode characters
        replacements = {
            # Smart quotes
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            # Dashes
            '—': '-',
            '–': '-',
            # Spaces
            '\u00A0': ' ',  # Non-breaking space
            '\u2009': ' ',  # Thin space
            '\u200A': ' ',  # Hair space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        if not self.config.remove_extra_whitespace:
            return text

        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove trailing/leading whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        
        if self.config.remove_empty_lines:
            # Remove empty lines
            lines = [line for line in lines if line]
        
        return '\n'.join(lines)

    def _remove_unwanted_patterns(self, text: str) -> str:
        """Remove unwanted patterns from text."""
        
        if self.config.remove_urls:
            # Remove URLs
            url_pattern = r'https?://[^\s]+'
            text = re.sub(url_pattern, '', text)
        
        if self.config.remove_emails:
            # Remove email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            text = re.sub(email_pattern, '', text)
        
        if self.config.remove_phone_numbers:
            # Remove phone numbers (simple pattern)
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
            text = re.sub(phone_pattern, '', text)
        
        if self.config.remove_code_blocks:
            # Remove code blocks (markdown style)
            code_pattern = r'```[\s\S]*?```|`[^`]+`'
            text = re.sub(code_pattern, '', text)
        
        if self.config.remove_special_chars:
            # Remove excessive special characters
            # Keep basic punctuation but remove repetitive chars
            text = re.sub(r'[^\w\s.,!?;:\'"()\[\]{}-]+', '', text)
            text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Max 2 repetitions
        
        return text

    def _filter_lines(self, text: str) -> str:
        """Filter lines based on length criteria."""
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_len = len(line.strip())
            
            # Skip lines that are too short or too long
            if line_len < self.config.min_line_length:
                continue
            if line_len > self.config.max_line_length:
                # Truncate very long lines
                line = line[:self.config.max_line_length] + '...'
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def _preserve_structure(self, text: str) -> str:
        """Preserve important document structure."""
        if not self.config.preserve_structure:
            return text

        # Preserve headers (markdown style)
        text = re.sub(r'^(#+\s+.+)$', r'\n\1\n', text, flags=re.MULTILINE)
        
        # Preserve list items
        text = re.sub(r'^(\s*[*+-]\s+.+)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*\d+\.\s+.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Ensure paragraph breaks
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()

    def preprocess_batch(self, documents: List[Document]) -> List[Document]:
        """Preprocess multiple documents."""
        preprocessed = []
        
        for doc in documents:
            processed_doc = self.preprocess(doc)
            preprocessed.append(processed_doc)
        
        return preprocessed

    def get_preprocessing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about preprocessing results."""
        if not documents:
            return {}

        total_docs = len(documents)
        original_lengths = []
        processed_lengths = []
        
        for doc in documents:
            if 'original_length' in doc.metadata:
                original_lengths.append(doc.metadata['original_length'])
                processed_lengths.append(doc.metadata['processed_length'])
        
        if not original_lengths:
            return {"error": "No preprocessing metadata found"}

        total_original = sum(original_lengths)
        total_processed = sum(processed_lengths)
        reduction_ratio = 1 - (total_processed / total_original) if total_original > 0 else 0

        return {
            "total_documents": total_docs,
            "total_original_chars": total_original,
            "total_processed_chars": total_processed,
            "reduction_ratio": reduction_ratio,
            "average_original_length": total_original / total_docs,
            "average_processed_length": total_processed / total_docs,
            "config": {
                "remove_extra_whitespace": self.config.remove_extra_whitespace,
                "remove_empty_lines": self.config.remove_empty_lines,
                "normalize_unicode": self.config.normalize_unicode,
                "min_line_length": self.config.min_line_length,
                "max_line_length": self.config.max_line_length,
            }
        }