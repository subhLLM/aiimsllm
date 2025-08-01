import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self,
                 bi_encoder_reranker='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 cross_encoder_reranker='cross-encoder/ms-marco-MiniLM-L-12-v2',
                 max_length=512):
        self.bi_encoder = None
        self.cross_encoder = None

        try:
            self.bi_encoder = CrossEncoder(bi_encoder_reranker, max_length=max_length)
            self.cross_encoder = CrossEncoder(cross_encoder_reranker, max_length=max_length)
            logger.info("Successfully loaded both bi-encoder and cross-encoder models.")
        except Exception as e:
            logger.error(f"Failed to load reranker models: {e}")

    def rerank_documents_bi_encoder(self, query, docs, top_k=8):
        if not self.bi_encoder:
            logger.warning("Bi-encoder not available. Skipping reranking.")
            return docs[:top_k]

        if not docs or len(docs) < 2:
            logger.info("Not enough documents to rerank.")
            return docs[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.bi_encoder.predict(pairs)
            scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            logger.info(f"Bi-encoder reranked {len(docs)} docs, returning top {top_k}.")
            return [doc for score, doc in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"Error in bi-encoder reranking: {e}")
            return docs[:top_k]

    def rerank_documents_cross_encoder(self, query, docs, top_k=3):
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available. Skipping reranking.")
            return docs[:top_k]

        if not docs or len(docs) < 2:
            logger.info("Not enough documents to rerank.")
            return docs[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            logger.info(f"Cross-encoder reranked {len(docs)} docs.")
            return [doc for score, doc in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return docs[:top_k]