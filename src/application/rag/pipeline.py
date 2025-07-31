"""RAG pipeline implementation."""

import time
from typing import List, Dict, Any, Optional

from src.core.interfaces import ILLMEngine, IVectorStore, Document
from src.core.exceptions import RAGPipelineError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self, llm_engine: ILLMEngine, vector_store: IVectorStore, config: Dict[str, Any]
    ):
        self.llm_engine = llm_engine
        self.vector_store = vector_store
        self.config = config

        # RAG settings
        self.top_k = config.get("top_k", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_context_length = config.get("max_context_length", 2000)
        self.reranking = config.get("reranking", True)

    async def process(
        self,
        query: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process query through RAG pipeline.

        Args:
            query: User query
            schema: Optional JSON schema for structured output
            max_tokens: Maximum tokens for generation
            temperature: LLM temperature

        Returns:
            Response dictionary with results and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            documents = await self._retrieve_documents(query)
            retrieval_time = int((time.time() - retrieval_start) * 1000)

            # Step 2: Build context
            context = self._build_context(documents)

            # Step 3: Create prompt
            prompt = self._create_prompt(query, context)

            # Step 4: Generate response
            generation_start = time.time()
            llm_response = await self.llm_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                schema=schema,
                **kwargs,
            )
            generation_time = int((time.time() - generation_start) * 1000)

            # Step 5: Parse and structure response
            result = self._parse_response(llm_response.text, schema)

            total_time = int((time.time() - start_time) * 1000)

            return {
                "result": result,
                "sources": [self._document_to_source(doc) for doc in documents],
                "tokens_used": llm_response.tokens_used,
                "model_name": llm_response.model_name,
                "confidence": self._calculate_confidence(documents),
                "metadata": {
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": generation_time,
                    "total_time_ms": total_time,
                    "documents_retrieved": len(documents),
                    "context_length": len(context),
                },
            }

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            raise RAGPipelineError(f"Pipeline failed: {e}")

    async def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        try:
            documents = await self.vector_store.search(
                query=query, top_k=self.top_k, threshold=self.similarity_threshold
            )

            # Rerank if enabled
            if self.reranking and documents:
                documents = await self._rerank_documents(query, documents)

            logger.debug(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            # Return empty list to allow generation without context
            return []

    async def _rerank_documents(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """Rerank documents based on relevance."""
        # Simple reranking based on keyword overlap
        query_terms = set(query.lower().split())

        scored_docs = []
        for doc in documents:
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0

            # Combine with similarity score
            final_score = (doc.similarity_score or 0) * 0.7 + score * 0.3
            scored_docs.append((final_score, doc))

        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs]

    def _build_context(self, documents: List[Document]) -> str:
        """Build context from documents."""
        if not documents:
            return ""

        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            # Format document
            doc_text = f"[Document {i+1} - {doc.source}]\n{doc.content}\n"

            # Check length
            if total_length + len(doc_text) > self.max_context_length:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM."""
        if context:
            prompt = f"""Based on the following context, please answer the query.

Context:
{context}

Query: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information, indicate what's missing."""
        else:
            prompt = f"""Please answer the following query:

Query: {query}

Please provide a comprehensive answer based on your knowledge."""

        return prompt

    def _parse_response(
        self, response_text: str, schema: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse and structure the response."""
        # If schema is provided, response should already be structured
        if schema:
            try:
                import json

                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse structured response")
                return {"answer": response_text}
        else:
            return {"answer": response_text}

    def _document_to_source(self, doc: Document) -> Dict[str, Any]:
        """Convert document to source info."""
        return {
            "content": doc.content[:200] + "..."
            if len(doc.content) > 200
            else doc.content,
            "source": doc.source,
            "chunk_id": doc.chunk_id,
            "similarity_score": doc.similarity_score or 0.0,
        }

    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents."""
        if not documents:
            return 0.5  # Medium confidence without context

        # Average similarity score of top documents
        scores = [doc.similarity_score or 0.0 for doc in documents[:3]]
        return sum(scores) / len(scores) if scores else 0.0
