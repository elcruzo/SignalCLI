"""Setup utilities for SignalCLI."""

import asyncio
import os
from pathlib import Path

from src.infrastructure.vector_store import create_vector_store
from src.infrastructure.embeddings.sentence_transformer import create_embedding_model
from src.utils.document_chunker import DocumentChunker
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def setup_vector_store():
    """Initialize vector store with sample documents."""
    config = load_config()

    # Create embedding model
    embedding_model = create_embedding_model(config["vector_store"])

    # Create vector store
    vector_store = create_vector_store(config["vector_store"], embedding_model)
    await vector_store.initialize()

    # Load sample documents
    sample_docs = [
        {
            "content": """Machine learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being explicitly programmed. 
            Machine learning focuses on the development of computer programs that can access data and use it 
            to learn for themselves.""",
            "metadata": {"source": "ml_basics.txt", "topic": "machine_learning"},
        },
        {
            "content": """Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a scripting 
            or glue language to connect existing components together.""",
            "metadata": {"source": "python_intro.txt", "topic": "programming"},
        },
        {
            "content": """Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
            computers understand, interpret and manipulate human language. NLP draws from many disciplines, 
            including computer science and computational linguistics, in its pursuit to fill the gap between 
            human communication and computer understanding.""",
            "metadata": {"source": "nlp_overview.txt", "topic": "nlp"},
        },
        {
            "content": """Deep learning is a subset of machine learning where artificial neural networks, 
            algorithms inspired by the human brain, learn from large amounts of data. Deep learning allows 
            machines to solve complex problems even when using a data set that is very diverse, unstructured 
            and inter-connected.""",
            "metadata": {"source": "deep_learning.txt", "topic": "deep_learning"},
        },
    ]

    # Chunk documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(sample_docs)

    # Add to vector store
    await vector_store.add_documents(chunks)

    logger.info(f"Added {len(chunks)} document chunks to vector store")
    await vector_store.shutdown()


async def verify_setup():
    """Verify SignalCLI setup."""
    checks = {
        "config": False,
        "models_dir": False,
        "logs_dir": False,
        "vector_store": False,
    }

    try:
        # Check config
        config = load_config()
        checks["config"] = True
    except Exception as e:
        logger.error(f"Config check failed: {e}")

    # Check directories
    if Path("models").exists():
        checks["models_dir"] = True
    else:
        Path("models").mkdir(exist_ok=True)

    if Path("logs").exists():
        checks["logs_dir"] = True
    else:
        Path("logs").mkdir(exist_ok=True)

    # Check vector store
    try:
        embedding_model = create_embedding_model(config["vector_store"])
        vector_store = create_vector_store(config["vector_store"], embedding_model)
        await vector_store.initialize()
        checks["vector_store"] = await vector_store.health_check()
        await vector_store.shutdown()
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")

    # Report results
    logger.info("Setup verification results:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}")

    return all(checks.values())


def main():
    """Main setup function."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        result = asyncio.run(verify_setup())
        sys.exit(0 if result else 1)
    else:
        asyncio.run(setup_vector_store())


if __name__ == "__main__":
    main()
