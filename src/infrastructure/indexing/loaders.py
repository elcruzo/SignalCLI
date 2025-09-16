"""Document loaders for various file types."""

import asyncio
import aiohttp
from pathlib import Path
from typing import Optional, Dict, Any
import mimetypes
import chardet

from src.core.interfaces import Document
from src.core.exceptions import DocumentLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Base document loader."""

    def __init__(self, max_file_size: int = 50 * 1024 * 1024):  # 50MB
        self.max_file_size = max_file_size
        self.loaders = {
            "text": TextLoader(),
            "web": WebLoader(),
            "pdf": PDFLoader(),
            "docx": DocxLoader(),
        }

    async def load_file(self, file_path: Path) -> Optional[Document]:
        """Load document from file."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.stat().st_size > self.max_file_size:
                raise DocumentLoadError(f"File too large: {file_path}")

            # Detect file type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            loader_type = self._get_loader_type(file_path, mime_type)

            # Load with appropriate loader
            loader = self.loaders.get(loader_type, self.loaders["text"])
            return await loader.load(file_path)

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    async def load_url(self, url: str) -> Optional[Document]:
        """Load document from URL."""
        try:
            return await self.loaders["web"].load(url)
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return None

    def _get_loader_type(self, file_path: Path, mime_type: Optional[str]) -> str:
        """Determine appropriate loader type."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf" or mime_type == "application/pdf":
            return "pdf"
        elif suffix in [".docx", ".doc"] or mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            return "docx"
        else:
            return "text"


class TextLoader:
    """Loads text-based files."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load text file."""
        try:
            # Read file with encoding detection
            content = await self._read_with_encoding(file_path)
            
            metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "modified_time": file_path.stat().st_mtime,
                "file_type": "text",
            }

            return Document(
                content=content,
                source=str(file_path),
                chunk_id=f"{file_path.name}#0",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return None

    async def _read_with_encoding(self, file_path: Path) -> str:
        """Read file with automatic encoding detection."""
        # Try to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            # Fallback encodings
            for fallback in ['utf-8', 'latin1', 'cp1252']:
                try:
                    return raw_data.decode(fallback)
                except UnicodeDecodeError:
                    continue
            
            # Last resort - ignore errors
            return raw_data.decode('utf-8', errors='ignore')


class WebLoader:
    """Loads content from web URLs."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def load(self, url: str) -> Optional[Document]:
        """Load content from URL."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"HTTP {response.status} for {url}")
                        return None

                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' in content_type:
                        content = await self._extract_html_text(await response.text())
                    else:
                        content = await response.text()

                    metadata = {
                        "url": url,
                        "content_type": content_type,
                        "status_code": response.status,
                        "file_type": "web",
                    }

                    return Document(
                        content=content,
                        source=url,
                        chunk_id=f"web#{hash(url)}",
                        metadata=metadata
                    )

        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return None

    async def _extract_html_text(self, html: str) -> str:
        """Extract text from HTML content."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text

        except ImportError:
            logger.warning("BeautifulSoup not available, returning raw HTML")
            return html
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            return html


class PDFLoader:
    """Loads PDF files."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load PDF file."""
        try:
            # Import PDF library
            try:
                import PyPDF2
            except ImportError:
                logger.error("PyPDF2 not installed, cannot load PDF files")
                return None

            content = await self._extract_pdf_text(file_path)
            
            metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "modified_time": file_path.stat().st_mtime,
                "file_type": "pdf",
            }

            return Document(
                content=content,
                source=str(file_path),
                chunk_id=f"{file_path.name}#0",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return None

    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF."""
        import PyPDF2
        
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        return "\n\n".join(text_parts)


class DocxLoader:
    """Loads Microsoft Word documents."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load DOCX file."""
        try:
            try:
                import python_docx
            except ImportError:
                logger.error("python-docx not installed, cannot load DOCX files")
                return None

            content = await self._extract_docx_text(file_path)
            
            metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "modified_time": file_path.stat().st_mtime,
                "file_type": "docx",
            }

            return Document(
                content=content,
                source=str(file_path),
                chunk_id=f"{file_path.name}#0",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return None

    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        import python_docx
        
        doc = python_docx.Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return "\n".join(text_parts)