import logging
import os
import shutil
from threading import Lock
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from doc_prepare import prepare_documents
from embedding import embedding_models, get_embedding_model_for_query
from config import FAISS_INDEX_PATH
from utils import detect_query_complexity

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, default_embedding_key="multilingual", index_path=FAISS_INDEX_PATH):
        self.embedding = embedding_models[default_embedding_key]
        self.index_path = index_path
        self.faiss_index_cache = {}
        self.bm25_model = None
        self.bm25_docs = []
        self.bm25_tokenized = []
        self._lock = Lock()

        self._initialize_faiss()
        self._initialize_bm25()

    def _initialize_faiss(self, embedding_instance=None):
        embedding_instance = embedding_instance or self.embedding
        model_id = embedding_instance.model_name

        if model_id in self.faiss_index_cache:
            logger.info(f"Using cached FAISS index for model: {model_id}")
            return self.faiss_index_cache[model_id]

        faiss_file = os.path.join(self.index_path, "index.faiss")
        pkl_file = os.path.join(self.index_path, "index.pkl")

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            try:
                logger.info(f"Loading FAISS index from {self.index_path} using model: {model_id}")
                db_faiss = FAISS.load_local(self.index_path, embedding_instance, allow_dangerous_deserialization=True)
                self.faiss_index_cache[model_id] = db_faiss
                return db_faiss
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Rebuilding.")
                shutil.rmtree(self.index_path, ignore_errors=True)
                os.makedirs(self.index_path, exist_ok=True)

        logger.info("Building new FAISS index for hospital data.")
        docs = prepare_documents()
        if not docs:
            logger.error("No documents prepared. FAISS index cannot be built.")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        try:
            db_faiss = FAISS.from_documents(chunks, embedding_instance)
            db_faiss.save_local(self.index_path)
            self.faiss_index_cache[model_id] = db_faiss
            logger.info(f"FAISS index built and saved with {len(chunks)} chunks.")
            return db_faiss
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return None

    def _initialize_bm25(self):
        documents = prepare_documents()
        if not documents:
            logger.error("No documents available for BM25 initialization.")
            return None

        self.bm25_docs = documents
        self.bm25_tokenized = [doc.page_content.lower().split() for doc in documents]

        if not self.bm25_tokenized:
            logger.error("BM25 tokenized corpus is empty.")
            return None

        self.bm25_model = BM25Okapi(self.bm25_tokenized)
        logger.info(f"Initialized BM25 with {len(self.bm25_tokenized)} documents.")
        return self.bm25_model

    def bm25_retrieve(self, query, k=5):
        if not self.bm25_model:
            logger.warning("BM25 not initialized. Returning empty list.")
            return []

        tokens = query.lower().split()
        scores = self.bm25_model.get_scores(tokens)
        ranked = sorted(zip(self.bm25_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def hybrid_search(self, query, k_simple=5, k_normal=8, k_complex=12):
        embedding_instance = get_embedding_model_for_query(query)
        faiss_db = self.faiss_index_cache.get(embedding_instance.model_name) or self._initialize_faiss(embedding_instance)

        if not faiss_db:
            logger.error("FAISS database not available.")
            return []

        if not self.bm25_model:
            logger.error("BM25 model not available. Using FAISS only.")
            return faiss_db.as_retriever(search_kwargs={"k": k_normal}).get_relevant_documents(query)

        complexity = detect_query_complexity(query)
        k_val = k_simple if complexity == "simple" else (k_normal if complexity == "normal" else k_complex)

        logger.info(f"Hybrid retrieval for '{query}' → complexity: {complexity}, k={k_val}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            faiss_future = executor.submit(faiss_db.as_retriever(search_kwargs={"k": k_val}).get_relevant_documents, query)
            bm25_future = executor.submit(self.bm25_retrieve, query, k_val)

            try:
                faiss_docs = faiss_future.result(timeout=10)
                bm25_docs = bm25_future.result(timeout=10)
            except TimeoutError:
                logger.warning("Retrieval timed out.")
                faiss_docs = faiss_future.result() if faiss_future.done() else []
                bm25_docs = bm25_future.result() if bm25_future.done() else []
            except Exception as e:
                logger.error(f"Error during retrieval: {e}")
                faiss_docs, bm25_docs = [], []

        merged = {doc.page_content: doc for doc in faiss_docs}
        for doc in bm25_docs:
            if doc.page_content not in merged:
                merged[doc.page_content] = doc

        final_docs = list(merged.values())
        logger.info(f"Hybrid merged {len(final_docs)} docs.")
        return final_docs[:k_val * 2]

    def refresh_indexes(self):
        with self._lock:
            logger.info("Refreshing FAISS and BM25 indexes.")
            self.faiss_index_cache.clear()
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path, ignore_errors=True)
            os.makedirs(self.index_path, exist_ok=True)
            self._initialize_faiss()
            self._initialize_bm25()
            logger.info("Indexes refreshed.")