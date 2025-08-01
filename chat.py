import logging
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from data_loader import HospitalDataLoader
from memory import ConversationMemory, InMemoryUserMemoryStore
from nlu import EnhancedNLUProcessor
from retriever import HybridRetriever
from reranker import DocumentReranker
from utils import (
    detect_conversational_intent, normalize_query, canonicalize_entity_value,
    generate_clarification_suggestions, format_doctor_availability, format_doctor_response,
    get_doctor_by_name, ensure_list, extract_doctor_name, detect_target_language_for_response,
    detect_and_translate, clean_extracted_entities, detect_query_complexity
)
from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

class HospitalChatbot:
    """Enhanced hospital chatbot with improved architecture and caching."""
    
    def __init__(self):
        # Initialize global dependencies
        self.data_loader = HospitalDataLoader()
        self.user_memory_store = InMemoryUserMemoryStore()
        self.nlu_processor = EnhancedNLUProcessor()
        self.retriever = HybridRetriever()
        self.reranker = DocumentReranker()
        
        # Cache for frequent queries
        self._response_cache = {}
        self._max_cache_size = 1000
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # Improved time-based greetings
        self.time_based_greetings = {
            'morning': ['Good morning', 'Morning'],
            'afternoon': ['Good afternoon', 'Afternoon'],
            'evening': ['Good evening', 'Evening'],
            'night': ['Hello', 'Hi']
        }
        
    def _get_cache_key(self, query: str, user_id: str) -> str:
        """Generate cache key for query."""
        return f"{user_id}:{hash(query.lower().strip())}"
    
    def _cache_response(self, key: str, response: dict) -> None:
        """Cache response with size limit."""
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }
    
    def _get_cached_response(self, key: str, max_age_minutes: int = 30) -> Optional[dict]:
        """Get cached response if still valid."""
        if key not in self._response_cache:
            return None
        
        cached_data = self._response_cache[key]
        age = (datetime.now() - cached_data['timestamp']).total_seconds() / 60
        
        if age > max_age_minutes:
            del self._response_cache[key]
            return None
        
        self.metrics['cache_hits'] += 1
        return cached_data['response']

    def classify_query_characteristics(self, query: str) -> Dict[str, str]:
        """Enhanced query classification with better patterns."""
        query_l = query.lower()
        response_length = "short"
        
        # More comprehensive patterns
        long_patterns = ["explain in detail", "everything about", "comprehensive", "thorough"]
        medium_patterns = ["list", "summarize", "overview of", "tell me about", "what are"]
        
        if any(pattern in query_l for pattern in long_patterns):
            response_length = "long"
        elif any(pattern in query_l for pattern in medium_patterns):
            response_length = "medium"
        
        return {"response_length": response_length}

    def detect_answer_style_and_tone(self, query: str) -> Tuple[str, str]:
        """Enhanced style and tone detection."""
        query_l = query.lower()
        style = "paragraph"
        tone = "professional_and_helpful"
        
        # Style detection patterns
        bullet_patterns = ["bullet points", "list them", "in bullets", "pointwise", 
                          "give points", "step by step", "in steps"]
        table_patterns = ["in a table", "tabular format", "as a table", "table format", 
                         "make a table", "structured table"]
        
        if any(pattern in query_l for pattern in bullet_patterns):
            style = "bullet_list"
        elif any(pattern in query_l for pattern in table_patterns):
            style = "table"
        
        # Tone detection patterns
        friendly_patterns = ["friendly", "casual", "informal", "talk like a friend", 
                           "light tone", "easy to understand", "simplify it"]
        formal_patterns = ["formal", "official statement", "strictly professional", 
                          "precise response", "in a formal tone", "business tone"]
        
        if any(pattern in query_l for pattern in friendly_patterns):
            tone = "friendly_and_casual"
        elif any(pattern in query_l for pattern in formal_patterns):
            tone = "formal_and_precise"
        
        return style, tone

    def rewrite_query_with_memory(self, query: str, memory: ConversationMemory) -> str:
        """Enhanced query rewriting with better coreference resolution."""
        original_query = query.strip()
        rewritten_query = original_query
        query_lower_normalized = normalize_query(original_query.lower())

        # Get context entities from recent conversation
        context_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
        
        # Find the most salient topic entity
        salient_topic_entity_value = memory.get_last_entity_by_priority(
            type_priority=["doctors", "departments", "services", "rooms", "hospitals", "buildings"]
        )
        salient_topic_type = None
        
        if salient_topic_entity_value:
            for entity_type in ["doctors", "departments", "services", "rooms", "hospitals", "buildings"]:
                if memory.last_entity_by_type.get(entity_type) == salient_topic_entity_value:
                    salient_topic_type = entity_type
                    break
        
        # Fallback to current topic or context entities
        if not salient_topic_entity_value and memory.current_topic:
            salient_topic_entity_value = memory.current_topic.get("value")
            salient_topic_type = memory.current_topic.get("type")
        elif not salient_topic_entity_value and context_entities:
            salient_topic_entity_value = context_entities[-1].get("value")
            salient_topic_type = context_entities[-1].get("type")

        # Enhanced follow-up pattern detection
        follow_up_keywords = [
            "contact", "email", "phone", "website", "location", "address", "services", 
            "specialty", "availability", "schedule", "timings", "hours", "visiting hours", 
            "profile", "about", "department", "room", "floor"
        ]
        
        follow_up_pattern_str = (
            r"^(and|then|also|what about|how about|tell me more about|more info on)\b.*(" + 
            "|".join(follow_up_keywords) + r")?|^(their|his|her|its)\b.*(" + 
            "|".join(follow_up_keywords) + r")|^\b(" + "|".join(follow_up_keywords) + r")\b"
        )
        
        is_short_follow_up_keyword_only = (
            len(original_query.split()) <= 2 and 
            any(kw in query_lower_normalized for kw in follow_up_keywords)
        )

        # Apply coreference resolution for follow-up queries
        if salient_topic_entity_value and (
            re.search(follow_up_pattern_str, query_lower_normalized, re.IGNORECASE) or 
            is_short_follow_up_keyword_only
        ):
            if not re.search(
                rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', 
                query_lower_normalized, 
                re.IGNORECASE
            ):
                rewritten_query = f"{salient_topic_entity_value} - {original_query}"
                logger.info(
                    f"[Coref Rewrite - Follow-up] Rewrote '{original_query}' → "
                    f"'{rewritten_query}' using salient entity '{salient_topic_entity_value}'"
                )
                return rewritten_query

        # Handle short queries with pronoun resolution
        if len(original_query.split()) < 5 and salient_topic_entity_value:
            if salient_topic_type in ["floors", "buildings"] or original_query.lower().startswith(
                ("what", "who", "where", "when", "why", "how")
            ):
                if rewritten_query != original_query:
                    logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
                return rewritten_query

            # Enhanced pronoun patterns
            pronoun_patterns = [
                r"\b(it)\b", r"\b(they)\b", r"\b(them)\b", 
                r"\b(their)\b(?!\s*(?:own|selves))",
                r"\b(his|her)\b",
                r"\b(its)\b",
                r"\b(this|that)\s*(one|department|doctor|room|service|place)?\b"
            ]
            
            for pattern in pronoun_patterns:
                match = re.search(pattern, rewritten_query, re.IGNORECASE)
                if match:
                    pronoun = match.group(0)
                    replacement_text = salient_topic_entity_value
                    if pronoun.lower() in ["his", "her", "their", "its"]:
                        replacement_text = f"{salient_topic_entity_value}'s"
                    
                    if not re.search(
                        rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', 
                        query_lower_normalized, 
                        re.IGNORECASE
                    ):
                        rewritten_query = re.sub(
                            pattern, replacement_text, rewritten_query, count=1, flags=re.IGNORECASE
                        )
                        logger.info(
                            f"[Coref Rewrite - Pronoun] Rewrote '{original_query}' → "
                            f"'{rewritten_query}' replacing '{pronoun}' with '{replacement_text}'"
                        )
                        break

        if rewritten_query != original_query:
            logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
        
        return rewritten_query

    def handle_small_talk(self, user_query: str, memory: ConversationMemory, user_id: str) -> Dict[str, str]:
        """Enhanced small talk handling with better context awareness."""
        current_hour = datetime.now().hour
        
        # Determine time-based greeting context
        if 5 <= current_hour < 12:
            time_context = random.choice(self.time_based_greetings['morning'])
        elif 12 <= current_hour < 17:
            time_context = random.choice(self.time_based_greetings['afternoon'])
        elif 17 <= current_hour < 21:
            time_context = random.choice(self.time_based_greetings['evening'])
        else:
            time_context = random.choice(self.time_based_greetings['night'])

        returning_user = hasattr(memory, "history") and len(memory.history) > 0
        
        # Enhanced conversational prompt
        convo_prompt = (
            f"You are AiimsBot, a warm and helpful AI assistant at AIIMS Jammu hospital.\n"
            f"Context: {time_context}. {'Returning visitor' if returning_user else 'New visitor'}.\n"
            f"The user said: '{user_query}'\n\n"
            f"Guidelines:\n"
            f"- Be naturally conversational and empathetic (hospital visitors may be stressed)\n"
            f"- Keep responses under 25 words but feel free to be warm\n"
            f"- Use appropriate emojis sparingly (1-2 max)\n"
            f"- If greeting, acknowledge the time of day naturally\n"
            f"- If goodbye, offer continued assistance\n"
            f"- Show you're ready to help with hospital-related queries\n"
            f"- Never repeat previous responses exactly\n"
            f"- Be sensitive to the hospital context - users might be anxious\n\n"
            f"Respond naturally (don't start with 'AiimsBot:'):"
        )

        try:
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.85)
            response = llm.invoke(convo_prompt).content.strip()
            # Clean response to remove any prefix
            response = response.replace("AiimsBot:", "").replace("Assistant:", "").strip()
        except Exception as e:
            logger.error(f"Error in small talk handler: {e}")
            # Enhanced context-aware fallbacks
            query_lower = user_query.lower()
            
            if any(word in query_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
                fallbacks = [
                    "Take care! Feel free to reach out anytime you need help. 👋",
                    "Goodbye! Hope your visit to AIIMS Jammu goes well! 🙏",
                    "Thank you! I'm always here if you need assistance. 😊",
                    "Wishing you well! Don't hesitate to ask if you need anything. 🌟"
                ]
            else:
                fallbacks = [
                    f"{time_context}! 👋 How can I assist you at AIIMS Jammu today?",
                    f"{time_context}! 😊 I'm here to help with anything hospital-related.",
                    f"{time_context}! Need directions, doctor info, or appointment help?",
                    f"{time_context}! Ready to help with any AIIMS Jammu questions! 🏥"
                ]
            
            response = random.choice(fallbacks)

        # Initialize memory if needed
        if not hasattr(memory, "history"):
            memory.history = []

        memory.add_turn(user_query, response, extracted_entities_map={})
        self.user_memory_store.save(user_id, memory)
        
        return {"answer": response}

    def _select_optimal_model(self, task_type: str, response_length_hint: str, query_complexity: str) -> Tuple[str, float]:
        """Select optimal model and temperature based on query characteristics."""
        if (response_length_hint == "long" or 
            task_type in ["explanation", "comparison", "listing_all"] or 
            "complex" in query_complexity):
            return "llama3-70b-8192", 0.4
        elif (task_type in ["contact_info", "location", "doctor_availability"] and 
              response_length_hint == "short"):
            return "llama3-8b-8192", 0.15
        else:
            return "llama3-70b-8192", 0.25

    def chat(self, user_query: str, user_id: str) -> Dict[str, Any]:
        """Main chat function with enhanced error handling and performance tracking."""
        request_start_time = datetime.now()
        self.metrics['total_queries'] += 1
        
        logger.info(f"--- New Chat Request (Hospital) --- User ID: {user_id} | Query: '{user_query}'")

        # Check cache first
        cache_key = self._get_cache_key(user_query, user_id)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info("Returning cached response")
            return cached_response

        try:
            conv_memory = self.user_memory_store.get(user_id)
            original_user_query = user_query.strip()
            query_lower_raw = original_user_query.lower()

            # Enhanced conversational intent detection
            convo_intent = detect_conversational_intent(original_user_query)
            hospital_entity_keywords = [
                "room", "opd", "icu", "ward", "doctor", "dr", "nurse", "staff", "physician", 
                "consultant", "specialist", "department", "cardiology", "neurology", "oncology", 
                "pediatrics", "radiology", "surgery", "clinic", "service", "x-ray", "mri", "scan", 
                "test", "appointment", "treatment", "procedure", "hospital", "aiims", "building", 
                "floor", "location", "find", "where", "contact", "phone", "email", "availability", 
                "schedule", "hours", "timings"
            ]
            
            # Handle pure conversational queries
            if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
                if not any(keyword in query_lower_raw for keyword in hospital_entity_keywords):
                    response = self.handle_small_talk(original_user_query, conv_memory, user_id)
                    self._cache_response(cache_key, response)
                    return response

            # Language detection and translation
            cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
            translated_query, detected_input_lang = detect_and_translate(cleaned_query_for_lang_detect)
            
            # Query rewriting with memory context
            processed_query = (
                self.rewrite_query_with_memory(translated_query, conv_memory)
                if len(translated_query.split()) < 7 and not any(
                    translated_query.lower().startswith(q_word) 
                    for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"]
                )
                else translated_query
            )

            # Enhanced entity extraction and canonicalization
            extracted_query_entities = self.nlu_processor.extract_entities(processed_query)
            extracted_query_entities = clean_extracted_entities(extracted_query_entities)

            # Canonicalize entity values
            for key, val in extracted_query_entities.items():
                if isinstance(val, str):
                    canon_val = canonicalize_entity_value(val)
                    if canon_val != val:
                        logger.info(f"[Canonicalization] '{val}' → '{canon_val}'")
                        extracted_query_entities[key] = canon_val
                elif isinstance(val, list):
                    new_list = []
                    for subval in val:
                        canon_sub = canonicalize_entity_value(subval)
                        if canon_sub != subval:
                            logger.info(f"[Canonicalization] '{subval}' → '{canon_sub}'")
                        new_list.append(canon_sub)
                    extracted_query_entities[key] = new_list

            # Tag matching
            known_tags = set(self.data_loader.get_all_metadata_tags())
            for key, val in extracted_query_entities.items():
                if isinstance(val, str) and val.lower() in known_tags:
                    logger.info(f"[Tag Match] Entity '{val}' matched known metadata tag")
                    extracted_query_entities[f"matched_tag__{key}"] = val.lower()
                elif isinstance(val, list):
                    for subval in val:
                        if isinstance(subval, str) and subval.lower() in known_tags:
                            logger.info(f"[Tag Match] List Entity '{subval}' matched known tag")
                            extracted_query_entities.setdefault(f"matched_tag__{key}", []).append(subval.lower())

            # Entity enrichment from memory
            if conv_memory and conv_memory.history:
                enriched_entities = extracted_query_entities.copy()
                important_entity_types = ["doctors", "departments", "rooms", "services", "buildings"]
                for ent_type in important_entity_types:
                    if not enriched_entities.get(ent_type):
                        last_value = conv_memory.get_last_entity_by_priority(ent_type)
                        if last_value:
                            logger.info(f"[Entity Fallback] Injected {ent_type}: {last_value}")
                            enriched_entities[ent_type] = [last_value]
                extracted_query_entities = enriched_entities

            # Task classification
            task_type = self.nlu_processor.classify_intent(processed_query)
            convo_intent = detect_conversational_intent(processed_query)

            # Handle mixed conversational + hospital queries
            greeting_resp = None
            if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
                if not any(word in query_lower_raw for word in hospital_entity_keywords):
                    response = self.handle_small_talk(original_user_query, conv_memory, user_id)
                    self._cache_response(cache_key, response)
                    return response
                else:
                    greeting_resp = self.handle_small_talk(original_user_query, conv_memory, user_id)["answer"]

            # Update conversation memory
            conv_memory.add_turn(original_user_query, "", extracted_query_entities)
            self.user_memory_store.save(user_id, conv_memory)
            logger.info(f"Detected task type (hospital): {task_type}")

            # Handle out-of-scope queries
            if task_type == "out_of_scope":
                out_of_scope_response = (
                    "I am an assistant for AIIMS Jammu hospital and can only answer questions "
                    "related to its facilities, departments, doctors, services, and appointments. "
                    "How can I help you with that?"
                )
                conv_memory.history[-1]["assistant"] = out_of_scope_response
                self.user_memory_store.save(user_id, conv_memory)
                
                # Translate if needed
                if target_lang_code and target_lang_code != "en":
                    out_of_scope_response = GoogleTranslator(
                        source="en", target=target_lang_code
                    ).translate(out_of_scope_response)
                elif detected_input_lang != "en":
                    out_of_scope_response = GoogleTranslator(
                        source="en", target=detected_input_lang
                    ).translate(out_of_scope_response)
                
                response = {"answer": out_of_scope_response, "debug_info": {"task_type": task_type}}
                self._cache_response(cache_key, response)
                return response

            # Check API key
            if not GROQ_API_KEY:
                logger.critical("Groq API key not configured.")
                return {"answer": "Error: Chat service temporarily unavailable."}

            # Query characteristics analysis
            query_chars = self.classify_query_characteristics(processed_query)
            response_length_hint = query_chars.get("response_length", "short")
            answer_style, answer_tone = self.detect_answer_style_and_tone(processed_query)
            logger.info(f"Response hints (hospital): length={response_length_hint}, style={answer_style}, tone={answer_tone}")

            # Document retrieval
            retrieved_docs = self.retriever.hybrid_search(
                processed_query, k_simple=6, k_normal=10, k_complex=15
            )
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for hospital query: {processed_query}")
                clarification_msg = "I couldn't find specific information for your query. "
                suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
                clarification_msg += " ".join(suggestions) if suggestions else "Could you try rephrasing or provide more details?"
                
                conv_memory.history[-1]["assistant"] = clarification_msg
                self.user_memory_store.save(user_id, conv_memory)
                
                # Translate if needed
                if target_lang_code and target_lang_code != "en":
                    clarification_msg = GoogleTranslator(
                        source="en", target=target_lang_code
                    ).translate(clarification_msg)
                elif detected_input_lang != "en":
                    clarification_msg = GoogleTranslator(
                        source="en", target=detected_input_lang
                    ).translate(clarification_msg)
                
                response = {
                    "answer": clarification_msg, 
                    "related_queries": suggestions if suggestions else []
                }
                self._cache_response(cache_key, response)
                return response

            # Document reranking
            bi_reranked_docs = self.reranker.rerank_documents_bi_encoder(
                processed_query, retrieved_docs, top_k=6
            )
            final_docs_for_llm = self.reranker.rerank_documents_cross_encoder(
                processed_query, bi_reranked_docs, top_k=3
            )

            # Doctor-specific handling
            doctor_candidates = extracted_query_entities.get("doctors", []) + extracted_query_entities.get("persons", [])
            doctor_candidates = [d for d in doctor_candidates if len(d.split()) >= 2]
            query_doctor_name = doctor_candidates[0] if doctor_candidates else extract_doctor_name(processed_query)

            if query_doctor_name:
                doctor_response = get_doctor_by_name(query_doctor_name, final_docs_for_llm)
                if doctor_response:
                    logger.info(f"[Doctor Match] Structured match found for: {query_doctor_name}")
                    response = {"answer": doctor_response}
                    self._cache_response(cache_key, response)
                    return response

            # Entity grounding check
            entity_terms_to_check = set()
            for ent_list in extracted_query_entities.values():
                for val in ent_list:
                    val_clean = val.lower().strip()
                    if val_clean and len(val_clean) > 1 and not val_clean.startswith("##"):
                        entity_terms_to_check.add(val_clean)

            logger.info(f"[Entity Grounding] Checking for terms in docs: {entity_terms_to_check}")

            missing_entities = []
            for term in entity_terms_to_check:
                found_in_docs = any(term in doc.page_content.lower() for doc in final_docs_for_llm)
                if not found_in_docs:
                    missing_entities.append(term)

            if missing_entities and task_type != "general_information":
                logger.warning(f"[Missing Context] Could not find these terms in retrieved docs: {missing_entities}")
                missing_list = ", ".join(missing_entities)
                clarification_msg = f"I couldn't find specific information about: {missing_list}. Could you clarify or rephrase your query?"
                
                conv_memory.history[-1]["assistant"] = clarification_msg
                self.user_memory_store.save(user_id, conv_memory)
                
                # Translate if needed
                if target_lang_code and target_lang_code != "en":
                    clarification_msg = GoogleTranslator(
                        source="en", target=target_lang_code
                    ).translate(clarification_msg)
                elif detected_input_lang != "en":
                    clarification_msg = GoogleTranslator(
                        source="en", target=detected_input_lang
                    ).translate(clarification_msg)
                
                response = {"answer": clarification_msg}
                self._cache_response(cache_key, response)
                return response

            # Inject top BM25 document if not already present
            top_bm25_docs_for_injection = self.retriever.bm25_retrieve(processed_query, k=1)
            if top_bm25_docs_for_injection:
                top_bm25_doc = top_bm25_docs_for_injection[0]
                if all(top_bm25_doc.page_content.strip() != doc.page_content.strip() for doc in final_docs_for_llm):
                    final_docs_for_llm.append(top_bm25_doc)
                    logger.info("Injected top BM25 doc into LLM context.")

            logger.info(f"Final {len(final_docs_for_llm)} documents selected for LLM context (hospital).")
            
            # Fallback if reranking resulted in zero documents
            if not final_docs_for_llm and retrieved_docs:
                final_docs_for_llm = retrieved_docs[:3]
                logger.warning("Reranking resulted in zero documents. Using top 3 from initial hybrid retrieval for LLM.")

            # Generate response using LLM
            final_response_text = self._generate_llm_response(
                processed_query, final_docs_for_llm, conv_memory, task_type, 
                answer_style, answer_tone, response_length_hint, extracted_query_entities
            )

            # Update conversation memory with response
            conv_memory.history[-1]["assistant"] = final_response_text
            
            # Add greeting if present
            if greeting_resp:
                final_response_text = f"{greeting_resp}\n\n{final_response_text}"

            self.user_memory_store.save(user_id, conv_memory)

            # Translate response if needed
            try:
                if target_lang_code and target_lang_code != "en":
                    final_response_text = GoogleTranslator(
                        source="en", target=target_lang_code
                    ).translate(final_response_text)
                    logger.info(f"Translated response to {target_lang_code}.")
                elif detected_input_lang != "en" and detected_input_lang is not None:
                    final_response_text = GoogleTranslator(
                        source="en", target=detected_input_lang
                    ).translate(final_response_text)
                    logger.info(f"Translated response back to input language {detected_input_lang}.")
            except Exception as e:
                logger.warning(f"Failed to translate final response: {e}")

            # Calculate processing time and update metrics
            processing_time = (datetime.now() - request_start_time).total_seconds()
            self._update_metrics(processing_time)
            
            logger.info(f"--- Chat Request Completed (Hospital) --- Time: {processing_time:.2f}s")
            
            # Prepare debug information
            debug_info = {
                "detected_task_type": task_type,
                "processed_query": processed_query,
                "detected_input_lang": detected_input_lang,
                "target_response_lang": target_lang_code,
                "answer_style": answer_style,
                "answer_tone": answer_tone,
                "response_length_hint": response_length_hint,
                "retrieved_docs_count_initial": len(retrieved_docs) if retrieved_docs else 0,
                "retrieved_docs_count_final_llm": len(final_docs_for_llm) if final_docs_for_llm else 0,
                "final_doc_ids_for_llm": [
                    doc.metadata.get("source_doc_id", "Unknown") 
                    for doc in final_docs_for_llm
                ] if final_docs_for_llm else [],
                "processing_time_seconds": round(processing_time, 2),
                "conversational_intent": convo_intent,
                "extracted_entities": extracted_query_entities,
                "query_complexity": self.classify_query_characteristics(processed_query),
                "missing_entities_in_docs": missing_entities,
                "cache_hit": False
            }
            
            response = {"answer": final_response_text, "debug_info": debug_info}
            self._cache_response(cache_key, response)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}", exc_info=True)
            error_response = {
                "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
                "debug_info": {"error": str(e)}
            }
            return error_response

    def _generate_llm_response(self, processed_query: str, final_docs_for_llm: List, 
                              conv_memory: ConversationMemory, task_type: str, 
                              answer_style: str, answer_tone: str, response_length_hint: str,
                              extracted_query_entities: Dict) -> str:
        """Generate LLM response with enhanced prompting."""
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(final_docs_for_llm):
            doc_text = f"Source Document {i+1}:\n{doc.page_content}\n"
            
            # Enhanced metadata extraction
            meta_info = {
                "Hospital": doc.metadata.get("hospital_name"),
                "Building": doc.metadata.get("building_name"),
                "Floor": doc.metadata.get("floor"),
                "Room Name": doc.metadata.get("room_name"),
                "Room Number": doc.metadata.get("room_number"),
                "Associated Depts": ", ".join(ensure_list(doc.metadata.get("associated_departments", []))[:2]),
                "Associated Doctors": ", ".join(ensure_list(doc.metadata.get("associated_doctors", []))[:2]),
                "Key Services": (
                    ", ".join(ensure_list(doc.metadata.get("services_directly_offered", []))[:2]) or
                    ", ".join(ensure_list(doc.metadata.get("department_related_services", []))[:2])
                ),
                "Doc ID": doc.metadata.get("source_doc_id")
            }
            
            filtered_meta_info = {k: v for k, v in meta_info.items() if v is not None and v != ""}
            if filtered_meta_info:
                doc_text += "Key Metadata: " + "; ".join([f"{k}: {v}" for k, v in filtered_meta_info.items()])
            
            context_parts.append(doc_text)
        
        extracted_context_str = "\n\n---\n\n".join(context_parts)

        # Enhanced prompt with task-specific instructions
        prompt_intro = (
            f"You are a highly advanced, intelligent, and conversational AI assistant for AIIMS Jammu Building. "
            f"Your primary goal is to provide accurate, concise, and relevant information based ONLY on the "
            f"'Extracted Context' provided. If the context is insufficient or irrelevant, clearly state that you "
            f"cannot answer or need more information. Do NOT invent information or use external knowledge."
        )

        # Task-specific instructions
        task_instructions = self._get_task_specific_instructions(task_type)

        # Build the complete prompt
        prompt_template_str = f"""{prompt_intro}

Strict Rules:
1. Base answers ONLY on 'Extracted Context'. If the information is not in the context, state that clearly 
   (e.g., "Based on the provided information, I cannot answer that," or "The context does not contain details about X."). 
   Do not use knowledge beyond this context. If multiple possible answers exist in the context, summarize them clearly. 
   If context is insufficient, say so politely.

2. If the Extracted Context is empty or clearly irrelevant to the query, state that you lack the necessary 
   information to answer.

3. Consider 'Past Conversation History' for resolving ambiguities (like "his email" referring to a previously 
   discussed doctor) but prioritize the current query and the 'Extracted Context' as the source of truth for the answer.

4. If the query is ambiguous despite context and history, you can ask ONE brief clarifying question.

5. Be conversational, empathetic, and helpful, adapting to a hospital setting where visitors may be anxious or stressed.

6. {task_instructions}

7. If asked about medical advice, conditions, or treatments, state that you are an AI assistant and cannot provide 
   medical advice. Suggest consulting with a healthcare professional. However, if the query is about *information 
   available in the context* regarding a service or procedure (e.g., "what does the context say about X-ray procedure?"), 
   then answer based on the context.

8. When possible, return structured answers:
   - Use **bullet points** for lists (e.g., multiple doctors, rooms, departments).
   - Use **labels** (e.g., Room Number: 301, Department: Radiology) to format details clearly.
   - For comparisons or listings, use a **table format** if relevant fields (name, location, contact, etc.) are available.
   - Avoid vague phrases like "at AIIMS Jammu" if room name, floor, and building info are present — include those explicitly.
   - If the answer refers to a specific person or entity mentioned earlier, restate the name for clarity 
     (e.g., "Dr. Aymen Masood is located in…").

9. Maintain a {answer_tone} tone and use {answer_style} format where appropriate.

10. Aim for a {response_length_hint} response length while ensuring completeness and accuracy.

Past Conversation History (Recent Turns):
{{history}}

Extracted Context (Source of Truth - Use ONLY this for answers):
---
{{context}}
---

User Query: {{input}}
Detected Task Type: {task_type}
Requested Answer Style: {answer_style}
Requested Tone: {answer_tone}
Desired Response Length: {response_length_hint}

Answer (provide only the answer, no preamble like "Here is the answer:"):
"""

        # Prepare input data for LLM
        chat_history_for_prompt = conv_memory.get_contextual_history_text(num_turns=4)
        llm_input_data = {
            "input": processed_query,
            "context": extracted_context_str,
            "history": chat_history_for_prompt,
            "task_type": task_type,
            "answer_style": answer_style,
            "answer_tone": answer_tone,
            "response_length_hint": response_length_hint
        }

        # Select optimal model
        query_complexity = detect_query_complexity(processed_query)
        groq_llm_model_name, temperature_val = self._select_optimal_model(
            task_type, response_length_hint, query_complexity
        )
        logger.info(f"Using Groq model (hospital): {groq_llm_model_name} with temperature: {temperature_val}")

        # Generate response
        llm = ChatGroq(api_key=GROQ_API_KEY, model=groq_llm_model_name, temperature=temperature_val)
        prompt = PromptTemplate.from_template(prompt_template_str)
        runnable_chain = prompt | llm
        
        final_response_text = "Error: Could not generate a response for your hospital query."
        try:
            ai_message = runnable_chain.invoke(llm_input_data)
            final_response_text = ai_message.content
            logger.info(f"LLM Raw Response Snippet (hospital): {final_response_text[:250]}...")
        except Exception as e:
            logger.error(f"Error invoking RAG chain with Groq (hospital): {e}")
            final_response_text = (
                "I apologize, but I encountered an issue while processing your request. "
                "The context might have been too large or there was a temporary service issue."
            )

        return final_response_text

    def _get_task_specific_instructions(self, task_type: str) -> str:
        """Get task-specific instructions for the LLM prompt."""
        instructions = {
            "location": (
                "When answering location-based queries, always provide clear and complete location details "
                "based ONLY on the Extracted Context. Include the hospital name, building name, zone/wing, "
                "floor number, and room number or name if present in the context. Avoid vague statements "
                "like 'located at AIIMS Jammu' unless that's all the context provides. If nearby landmarks "
                "or access points (like lifts, stairs, or entrances) are mentioned, include them too. "
                "Be precise, structured, and helpful."
            ),
            "location_specific": (
                "Provide specific location details including building, floor, room number, and any "
                "directional information available in the context."
            ),
            "location_general": (
                "Provide general location information and context about the area or facility."
            ),
            "contact_info": (
                "Extract and provide specific contact details like email, phone numbers, or website URLs "
                "for the queried entity (hospital, department, doctor) from the context. If multiple "
                "contacts exist, list them clearly."
            ),
            "operating_hours": (
                "Clearly state the operating hours, availability, days of the week, start, and end times "
                "as found in the context for the queried entity (e.g., OPD, doctor, service)."
            ),
            "doctor_availability": (
                "Provide specific doctor availability information including days, times, and any special "
                "scheduling notes from the context."
            ),
            "explanation": (
                "Provide a comprehensive explanation or description based on the context. If the context "
                "has a summary for a room or service, use it but elaborate with other details if available."
            ),
            "general_information": (
                "Provide general information about the topic based on what's available in the context."
            ),
            "department_info": (
                "Describe the department including its services, location, staff, and any special features "
                "mentioned in the context."
            ),
            "service_info": (
                "Explain what the service offers, how to access it, any requirements, and relevant details "
                "from the context."
            ),
            "listing_all": (
                "List all relevant items (e.g., doctors in a department, services offered, rooms on a floor) "
                "based on the query and context. Use bullet points for clarity."
            ),
            "listing_specific": (
                "List specific items that match the query criteria, organizing them clearly."
            ),
            "booking_info": (
                "Provide details on how to book an appointment or access a service, including method, "
                "contact for booking, or relevant URLs if found in the context. Mention if approval is required."
            ),
            "comparison": (
                "Compare the relevant entities (e.g., doctors, services, treatments) based on the information "
                "available in the context, highlighting differences and similarities in aspects like specialty, "
                "availability, or features."
            )
        }
        
        return instructions.get(task_type, "Provide accurate information based on the context.")

    def _update_metrics(self, processing_time: float) -> None:
        """Update performance metrics."""
        # Update average response time
        total_queries = self.metrics['total_queries']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_hit_rate = (
            self.metrics['cache_hits'] / self.metrics['total_queries'] * 100
            if self.metrics['total_queries'] > 0 else 0
        )
        
        return {
            'total_queries': self.metrics['total_queries'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'average_response_time_seconds': round(self.metrics['average_response_time'], 2),
            'cache_size': len(self._response_cache)
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._response_cache.clear()
        logger.info("Response cache cleared")

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        logger.info("Performance metrics reset")


# Global instance for backward compatibility
chatbot_instance = HospitalChatbot()

# Global functions for backward compatibility
def classify_query_characteristics(query):
    return chatbot_instance.classify_query_characteristics(query)

def detect_answer_style_and_tone(query):
    return chatbot_instance.detect_answer_style_and_tone(query)

def rewrite_query_with_memory(query, memory):
    return chatbot_instance.rewrite_query_with_memory(query, memory)

def handle_small_talk(user_query, memory, user_id):
    return chatbot_instance.handle_small_talk(user_query, memory, user_id)

def chat(user_query: str, user_id: str):
    """Main chat function - backward compatible interface."""
    return chatbot_instance.chat(user_query, user_id)