import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

embedding_models = {
    "multilingual": HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "qa": HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"),
    "general": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "ranking": HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5"),
    "hybrid": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
    "factual": HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2"),
}

def get_embedding_model_for_query(query):
    query_lower = query.lower()
    if any(word in query_lower for word in ["search", "find", "nearby", "available", "locate", "which room", "what rooms", "where can i find", "provide"]):
        return embedding_models["hybrid"]
    if any(word in query_lower for word in ["what is", "who is", "where is", "define", "tell me about", "doctor", "department", "opd", 'office']):
        return embedding_models["qa"]
    if any(word in query_lower for word in ["explain", "describe", "details about", "how does", "procedure", "treatment", "everything about"]):
        return embedding_models["factual"]
    if any(word in query_lower for word in ["list all", "compare services", "rank doctors", "compare", "best", "vs", "versus", "list of some", "list of any", "list of five", "list of top", "table"]):
        return embedding_models["ranking"]
    model = embedding_models["multilingual"]
    
    logger.info(f"[Embedding Model Routing] Using: {model.model_name} for query: {query}")
    return model