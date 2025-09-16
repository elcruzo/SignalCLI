"""Document indexer for building vector databases."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from src.core.interfaces import Document, IVectorStore, IEmbeddingModel
from src.core.exceptions import IndexingError
from .chunker import DocumentChunker, ChunkConfig
from .loaders import DocumentLoader
from .preprocessor import DocumentPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIndexer:
    """Indexes documents into vector store for RAG."""

    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_model: IEmbeddingModel,
        chunk_config: Optional[ChunkConfig] = None,
        max_workers: int = 4,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = DocumentChunker(chunk_config or ChunkConfig())
        self.loader = DocumentLoader()
        self.preprocessor = DocumentPreprocessor()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def index_file(self, file_path: Union[str, Path]) -> int:
        """
        Index a single file.

        Args:
            file_path: Path to file to index

        Returns:
            Number of chunks indexed
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Indexing file: {file_path}")

            # Load document
            document = await self.loader.load_file(file_path)
            if not document:
                logger.warning(f"Failed to load file: {file_path}")
                return 0

            # Process document
            processed_doc = await self.preprocessor.preprocess(document)

            # Chunk document
            chunks = self.chunker.chunk_text(processed_doc.content, str(file_path))

            # Generate embeddings and index
            return await self._index_chunks(chunks)

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            raise IndexingError(f"Failed to index file: {e}")

    async def index_directory(
        self,
        directory_path: Union[str, Path],
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> int:
        """
        Index all files in a directory.

        Args:
            directory_path: Path to directory
            patterns: File patterns to include (e.g., ["*.txt", "*.md"])
            recursive: Whether to search subdirectories

        Returns:
            Total number of chunks indexed
        """
        try:
            directory_path = Path(directory_path)
            if not directory_path.is_dir():
                raise ValueError(f"Not a directory: {directory_path}")

            logger.info(f"Indexing directory: {directory_path}")

            # Find files to index
            files = self._find_files(directory_path, patterns, recursive)
            logger.info(f"Found {len(files)} files to index")

            # Index files in parallel
            tasks = [self.index_file(file_path) for file_path in files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful indexing
            total_chunks = 0
            errors = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to index {files[i]}: {result}")
                    errors += 1
                else:
                    total_chunks += result

            logger.info(
                f"Directory indexing complete: {total_chunks} chunks, {errors} errors"
            )
            return total_chunks

        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            raise IndexingError(f"Failed to index directory: {e}")

    async def index_text(self, text: str, source: str = "text_input") -> int:
        """
        Index raw text content.

        Args:
            text: Text content to index
            source: Source identifier

        Returns:
            Number of chunks indexed
        """
        try:
            logger.info(f"Indexing text from source: {source}")

            # Create document
            document = Document(
                content=text,
                source=source,
                chunk_id=f"{source}#0",
                metadata={"indexed_at": asyncio.get_event_loop().time()},
            )

            # Preprocess
            processed_doc = await self.preprocessor.preprocess(document)

            # Chunk
            chunks = self.chunker.chunk_text(processed_doc.content, source)

            # Index
            return await self._index_chunks(chunks)

        except Exception as e:
            logger.error(f"Error indexing text: {e}")
            raise IndexingError(f"Failed to index text: {e}")

    async def index_url(self, url: str) -> int:
        """
        Index content from a URL.

        Args:
            url: URL to index

        Returns:
            Number of chunks indexed
        """
        try:
            logger.info(f"Indexing URL: {url}")

            # Load from URL
            document = await self.loader.load_url(url)
            if not document:
                logger.warning(f"Failed to load URL: {url}")
                return 0

            # Process and index
            processed_doc = await self.preprocessor.preprocess(document)
            chunks = self.chunker.chunk_text(processed_doc.content, url)

            return await self._index_chunks(chunks)

        except Exception as e:
            logger.error(f"Error indexing URL {url}: {e}")
            raise IndexingError(f"Failed to index URL: {e}")

    async def _index_chunks(self, chunks: List[Document]) -> int:
        """Index document chunks with embeddings."""
        if not chunks:
            return 0

        try:
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")

            # Generate embeddings in batches
            batch_size = 32
            batches = [
                chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
            ]

            indexed_count = 0

            for batch in batches:
                # Extract text for embedding
                texts = [doc.content for doc in batch]

                # Generate embeddings
                embeddings = await self.embedding_model.encode(texts)

                # Add embeddings to documents
                for doc, embedding in zip(batch, embeddings):
                    doc.embedding = embedding

                # Store in vector database
                await self.vector_store.add_documents(batch)
                indexed_count += len(batch)

            logger.info(f"Successfully indexed {indexed_count} chunks")
            return indexed_count

        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise IndexingError(f"Failed to index chunks: {e}")

    def _find_files(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Find files matching patterns."""
        files = []

        if patterns is None:
            # Default patterns for common text files
            patterns = [
                "*.txt",
                "*.md",
                "*.rst",
                "*.py",
                "*.js",
                "*.html",
                "*.css",
                "*.json",
                "*.yaml",
                "*.yml",
                "*.csv",
                "*.xml",
                "*.tex",
            ]

        if recursive:
            search_pattern = "**/*"
        else:
            search_pattern = "*"

        for pattern in patterns:
            for file_path in directory.glob(f"{search_pattern}"):
                if file_path.is_file() and file_path.match(pattern):
                    files.append(file_path)

        # Remove duplicates and sort
        files = sorted(list(set(files)))
        return files

    async def reindex_source(self, source: str) -> int:
        """
        Reindex all chunks from a source.

        Args:
            source: Source identifier to reindex

        Returns:
            Number of chunks reindexed
        """
        try:
            logger.info(f"Reindexing source: {source}")

            # Remove existing chunks from this source
            await self.vector_store.delete_by_source(source)

            # Reindex based on source type
            if source.startswith("http"):
                return await self.index_url(source)
            elif Path(source).exists():
                if Path(source).is_file():
                    return await self.index_file(source)
                else:
                    return await self.index_directory(source)
            else:
                logger.warning(f"Cannot reindex unknown source type: {source}")
                return 0

        except Exception as e:
            logger.error(f"Error reindexing source {source}: {e}")
            raise IndexingError(f"Failed to reindex source: {e}")

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        try:
            # Get vector store stats
            store_stats = await self.vector_store.get_stats()

            return {
                "total_documents": store_stats.get("document_count", 0),
                "total_vectors": store_stats.get("vector_count", 0),
                "index_size": store_stats.get("index_size", 0),
                "embedding_model": self.embedding_model.model_name,
                "chunk_strategy": self.chunker.config.strategy.value,
                "chunk_size": self.chunker.config.chunk_size,
                "vector_store": self.vector_store.__class__.__name__,
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup indexer resources."""
        self.executor.shutdown(wait=True)
        logger.info("Document indexer cleaned up")
