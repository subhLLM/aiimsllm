from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import logging
import os
import re
import shutil
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import numpy as np
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline, AutoModel
import torch
from collections import Counter
import sys
import io
import threading
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from rapidfuzz import fuzz
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from functools import lru_cache
import hashlib
from cursor import move_cursor
move_cursor()
from langchain.schema import HumanMessage, AIMessage
import redis
from collections import defaultdict
from cachetools import TTLCache
import time


# Full list of Indian languages supported by Sarvam-Translate
INDIC_LANGS = {
    "as", "bn", "brx", "doi", "gu", "en", "hi", "kn", "ks", "kok",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
}
# Load Sarvam model once
sarvam_model = AutoModel.from_pretrained("sarvamai/sarvam-translate")
sarvam_tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-translate")

class TranslationManager:
    def __init__(self):
        self.tokenizer = sarvam_tokenizer
        self.model = sarvam_model
        self.indic_langs = INDIC_LANGS

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            logger.warning(f"[LangDetect] Failed: {e}")
            return "en"

    def sarvam_translate(self, text, src_lang="hin", tgt_lang="eng"):
        try:
            input_text = f"{src_lang}>>{tgt_lang}>>{text}"
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
            outputs = self.model.generate(**inputs, max_length=512)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"[Sarvam] Failed: {e}")
            return text

    def deep_translate(self, text, source="auto", target="en"):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception as e:
            logger.warning(f"[DeepTranslator] Failed: {e}")
            return text

    def translate_to_english(self, text):
        lang = self.detect_language(text)
        if lang == "en":
            return text, "en"
        if lang in self.indic_langs:
            translated = self.sarvam_translate(text, src_lang=lang, tgt_lang="eng")
        else:
            translated = self.deep_translate(text, source=lang, target="en")
        return translated, lang

    def translate_from_english(self, text, target_lang):
        if target_lang in self.indic_langs:
            return self.sarvam_translate(text, src_lang="eng", tgt_lang=target_lang)
        else:
            return self.deep_translate(text, source="en", target=target_lang)

# Initialize hybrid translator (Sarvam + Deep Translator)
translator_manager = TranslationManager()

console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log", encoding='utf-8'),
        logging.StreamHandler(console_stream)
    ]
)
logger = logging.getLogger(__name__)

# --- Load spaCy model ---
try:
    nlp_spacy = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy en_core_web_lg successfully.")
    phrase_matcher = Matcher(nlp_spacy.vocab)
except Exception as e:
    logger.warning(f"Failed to load en_core_web_lg: {e}")
    nlp_spacy = None
    phrase_matcher = None


# === NLU PROCESSOR ===
class NLUProcessor:
    MODEL_CONFIG = {
        "intent_encoder": "all-mpnet-base-v2",
        "zero_shot": "joeddav/xlm-roberta-large-xnli",
        "ner_primary": {
            "model": "dslim/bert-base-NER",
            "tokenizer": "dslim/bert-base-NER",
            "use_fast": True
        },
        "ner_fallback": {
            "model": "Davlan/xlm-roberta-base-ner-hrl",
            "tokenizer": "Davlan/bert-base-multilingual-cased-ner-hrl",
            "use_fast": True
        },
        "ner_indic": {
            "model": "ai4bharat/IndicNER",
            "tokenizer": "ai4bharat/IndicNER",
            "use_fast": False
        }
    }

    def __init__(self):
        self.intent_labels, self.intent_metadata = self.load_intent_labels_metadata()
        self._init_medical_terms()
        
        # Initialize model placeholders
        self._intent_encoder = None
        self._intent_label_embeddings = None
        self._zero_shot_classifier = None
        self._ner_pipeline_primary = None
        self._ner_pipeline_fallback = None
        self._ner_pipeline_indic = None

    @property
    def intent_encoder(self):
        if self._intent_encoder is None:
            self._intent_encoder = SentenceTransformer(self.MODEL_CONFIG["intent_encoder"])
        return self._intent_encoder

    @property
    def intent_label_embeddings(self):
        if self._intent_label_embeddings is None:
            self._intent_label_embeddings = self.intent_encoder.encode(self.intent_labels, convert_to_tensor=True)
        return self._intent_label_embeddings

    @property
    def zero_shot_classifier(self):
        if self._zero_shot_classifier is None:
            zs_model_name = self.MODEL_CONFIG["zero_shot"]
            self._zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=AutoModelForSequenceClassification.from_pretrained(zs_model_name),
                tokenizer=AutoTokenizer.from_pretrained(zs_model_name, use_fast=False)
            )
        return self._zero_shot_classifier

    @property
    def ner_pipeline_primary(self):
        if self._ner_pipeline_primary is None:
            ner_primary_config = self.MODEL_CONFIG["ner_primary"]
            self._ner_pipeline_primary = pipeline(
                "ner", 
                model=ner_primary_config["model"], 
                tokenizer=AutoTokenizer.from_pretrained(ner_primary_config["tokenizer"], use_fast=ner_primary_config["use_fast"]),
                aggregation_strategy="simple"
            )
        return self._ner_pipeline_primary

    @property
    def ner_pipeline_fallback(self):
        if self._ner_pipeline_fallback is None:
            ner_fallback_config = self.MODEL_CONFIG["ner_fallback"]
            self._ner_pipeline_fallback = pipeline(
                "ner", 
                model=AutoModelForTokenClassification.from_pretrained(ner_fallback_config["model"]), 
                tokenizer=AutoTokenizer.from_pretrained(ner_fallback_config["tokenizer"], use_fast=ner_fallback_config["use_fast"]),
                aggregation_strategy="simple"
            )
        return self._ner_pipeline_fallback

    @property
    def ner_pipeline_indic(self):
        if self._ner_pipeline_indic is None:
            ner_indic_config = self.MODEL_CONFIG["ner_indic"]
            self._ner_pipeline_indic = pipeline(
                "ner",
                model=AutoModelForTokenClassification.from_pretrained(ner_indic_config["model"]),
                tokenizer=AutoTokenizer.from_pretrained(ner_indic_config["tokenizer"], use_fast=ner_indic_config["use_fast"]),
                aggregation_strategy="simple"
            )
        return self._ner_pipeline_indic

    def get_intent_labels(self):
        return [
            "location", "get_directions", "contact_info", "doctor_availability", "get_doctor_details",
            "booking_info", "operating_hours", "department_info", "service_info", "explanation",
            "general_information", "listing_all", "listing_specific", "comparison",
            "emergency_info", "accessibility_info", "how_to_question", "definition_question",
            "out_of_scope"
        ]
    
    def load_intent_labels_metadata(self):
        labels = self.get_intent_labels()
        metadata = {label: {"description": f"Intent: {label}"} for label in labels}
        return labels, metadata
    
    def _init_medical_terms(self):
        """Initialize medical terminology variations"""
        self.medical_term_variations = {
            # Personnel
            "doctor": ["dr", "doctor", "physician", "specialist", "consultant", "prof", "professor"],
            "nurse": ["nurse", "nursing staff", "sister", "staff nurse"],
            "staff": ["staff", "personnel", "employee", "worker"],
            
            # Departments
            "department": ["dept", "department", "unit", "wing", "division", "center", "centre"],
            "ward": ["ward", "room", "chamber", "cabin", "bay"],
            "clinic": ["clinic", "opd", "outpatient", "consultation"],
            
            # Facilities
            "hospital": ["hospital", "medical center", "healthcare center", "medical facility"],
            "emergency": ["emergency", "urgent care", "casualty", "trauma", "er", "accident"],
            "icu": ["icu", "intensive care", "critical care", "ccu"],
            
            # Services
            "test": ["test", "investigation", "examination", "analysis"],
            "scan": ["scan", "imaging", "x-ray", "radiograph"],
            "procedure": ["procedure", "operation", "surgery", "intervention"],
            
            # Administrative
            "appointment": ["appointment", "booking", "scheduling", "consultation"],
            "admission": ["admission", "hospitalization", "inpatient", "registration"],
            "payment": ["payment", "bill", "fee", "charge", "cost"]
        }
        
        # Build reverse lookup for normalization
        self.term_normalization = {}
        for standard, variations in self.medical_term_variations.items():
            for variant in variations:
                self.term_normalization[variant] = standard
    
    def extract_entities(self, query: str) -> dict:
        """Enhanced entity extraction with a streamlined pipeline."""
        try:
            entities = defaultdict(list)

            # Run the NER pipeline
            self._run_ner_pipeline(query, entities)

            # Apply regex patterns for additional entity extraction
            self._apply_regex_patterns(query, entities)

            # Normalize medical terms to standardize entity values
            self._normalize_medical_terms(query, entities)

            # Validate and clean the final list of entities
            cleaned_entities = self._validate_and_clean_entities(entities)
            
            return dict(cleaned_entities)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def _run_ner_pipeline(self, query: str, entities: dict):
        """Sequentially run NER models until entities are found."""
        # Try primary NER model (English-focused)
        if self.ner_pipeline_primary:
            primary_results = self.ner_pipeline_primary(query)
            if primary_results:
                self._process_ner_results(primary_results, entities, source="primary")
                return

        # If primary fails or finds nothing, try the fallback model
        if self.ner_pipeline_fallback:
            fallback_results = self.ner_pipeline_fallback(query)
            if fallback_results:
                self._process_ner_results(fallback_results, entities, source="fallback")
                return

        # If both primary and fallback fail, try the Indic NER model
        if self.ner_pipeline_indic:
            indic_results = self.ner_pipeline_indic(query)
            if indic_results:
                self._process_ner_results(indic_results, entities, source="indic")
    
    def _process_ner_results(self, results: list, entities: dict, source: str):
        """Process NER results and map to entity types"""
        ner_mapping = {
            "PER": "doctors",
            "ORG": "departments",
            "LOC": "locations",
            "DATE": "dates",
            "TIME": "times",
            "MISC": "misc"
        }
        
        for result in results:
            entity_type = result.get("entity_group")
            if entity_type in ner_mapping:
                mapped_type = ner_mapping[entity_type]
                value = result.get("word").strip()
                if value and len(value) > 1:  # Basic validation
                    entities[mapped_type].append({
                        "value": value,
                        "source": source,
                        "confidence": result.get("score", 1.0)
                    })
    
    def _apply_regex_patterns(self, query: str, entities: dict):
        """Apply regex patterns for entity extraction"""
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                if value:
                    entities[entity_type].append({
                        "value": value,
                        "source": "regex",
                        "confidence": 1.0
                    })
    
    def _normalize_medical_terms(self, query: str, entities: dict):
        """Normalize medical terms and add to entities"""
        words = query.lower().split()
        for i, word in enumerate(words):
            if word in self.term_normalization:
                standard_term = self.term_normalization[word]
                # Look for multi-word matches
                for j in range(i+1, min(i+4, len(words))):
                    phrase = " ".join(words[i:j])
                    if phrase in self.term_normalization:
                        standard_term = self.term_normalization[phrase]
                        break
                entities[standard_term].append({
                    "value": word,
                    "source": "medical_terms",
                    "confidence": 1.0
                })
    
    def _validate_and_clean_entities(self, entities: dict) -> dict:
        """Validate and clean extracted entities"""
        cleaned = defaultdict(list)
        for entity_type, values in entities.items():
            # Remove duplicates while preserving order
            seen = set()
            for value in values:
                if isinstance(value, dict):
                    val_str = value["value"]
                else:
                    val_str = str(value)
                
                if val_str not in seen and len(val_str.strip()) > 1:
                    seen.add(val_str)
                    cleaned[entity_type].append(value)
        
        return dict(cleaned)
    
    def classify_intent(self, query: str) -> tuple:
        """Enhanced intent classification with confidence"""
        try:
            if not self.intent_classifier:
                return self._rule_based_intent(query), 1.0
            
            # Zero-shot classification
            result = self.intent_classifier(
                query,
                candidate_labels=self.intent_labels,
                hypothesis_template="This is a query about {}.",
                multi_label=False
            )
            
            intent = result["labels"][0]
            confidence = result["scores"][0]
            
            # Apply rule-based corrections for high-confidence cases
            rule_based_intent = self._rule_based_intent(query)
            if rule_based_intent != "general_information":
                return rule_based_intent, 0.95
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._rule_based_intent(query), 0.7
    
    def _rule_based_intent(self, query: str) -> str:
        """Rule-based intent classification fallback based on keywords."""
        query_lower = query.lower()

        # Check for location-related intents
        if self._is_location_intent(query_lower):
            return self._get_location_intent_type(query_lower)

        # Check for other specific intents
        if any(term in query_lower for term in ["book", "schedule", "appointment", "reserve"]):
            return "appointment_booking"
        if any(term in query_lower for term in ["contact", "phone", "email", "call"]):
            return "contact_info"
        if any(term in query_lower for term in ["time", "hour", "open", "close", "schedule"]):
            return "operating_hours"
        if any(term in query_lower for term in ["list", "show all", "available"]):
            return "listing_specific" if any(term in query_lower for term in ["doctor", "staff", "specialist"]) else "listing_all"
        if any(term in query_lower for term in ["compare", "difference", "better"]):
            return "comparison"
        if any(term in query_lower for term in ["explain", "how does", "what is", "tell me about"]):
            return "explanation"
        if any(term in query_lower for term in ["emergency", "urgent", "immediate"]):
            return "emergency_info"

        # Default to general information if no other intent is matched
        return "general_information"

    def _is_location_intent(self, query_lower: str) -> bool:
        """Check if the query is related to finding a location."""
        return any(term in query_lower for term in ["where is", "how to reach", "locate", "find"])

    def _get_location_intent_type(self, query_lower: str) -> str:
        """Determine the specific type of location-related intent."""
        if any(term in query_lower for term in ["room", "ward"]):
            return "find_facility"
        if any(term in query_lower for term in ["department", "unit", "center"]):
            return "find_department"
        if any(term in query_lower for term in ["doctor", "dr", "physician"]):
            return "find_person"
        return "location"

    def classify_intent_with_fallback(self, query: str) -> tuple:
        """
        Classify intent using zero-shot, fallback to LLM if confidence is low.
        """
        intent, confidence = self.classify_intent(query)
        if confidence < 0.6 or intent in {"general_information", "out_of_scope"}:
            try:
                # Use LLM for fallback intent detection
                system_prompt = (
                    "You are a smart assistant. Classify the following user query into one of these intents: "
                    + ", ".join(self.intent_labels) + ".\n"
                    "Return only the intent label."
                )
                llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.1)
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ])
                llm_intent = response.content.strip()
                if llm_intent in self.intent_labels:
                    return llm_intent, 0.8
            except Exception as e:
                logger.warning(f"LLM fallback intent detection failed: {e}")
        return intent, confidence

    def extract_entities_with_fallback(self, query: str) -> dict:
        """
        Extract entities using all available methods, fallback to LLM if empty.
        """
        entities = self.extract_entities(query)
        if not any(entities.values()):
            try:
                # Use LLM for fallback entity extraction
                system_prompt = (
                    "Extract all relevant entities from the following user query. "
                    "Return a JSON object with keys for each entity type (e.g., doctor, department, certificate_type, complaint, feedback, gov_service, etc.). "
                    "If no entity is found for a type, use an empty list."
                )
                llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.1)
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ])
                import json
                llm_entities = json.loads(response.content)
                if isinstance(llm_entities, dict):
                    return llm_entities
            except Exception as e:
                logger.warning(f"LLM fallback entity extraction failed: {e}")
        return entities


# === CONVERSATION MEMORY MODULE ===
class ConversationMemory:
    CONFIG = {
        "max_history_turns": 10,
        "summary_threshold": 15,
        "prune_keep_count": 9,
        "context_max_turns": 5,
        "context_relevance_threshold": 0.3,
        "recent_satisfaction_window_seconds": 300,
        "similar_query_threshold": 0.8
    }

    def __init__(self, max_history_turns=None, summary_threshold=None):
        # Core conversation storage
        self.history = []
        self.contextual_entities = TTLCache(maxsize=100, ttl=3600)  # Cache entities for 1 hour
        self.current_topic = None
        self.last_entity_by_type = {}
        
        # Configuration
        self.max_history_turns = max_history_turns or self.CONFIG["max_history_turns"]
        self.summary_threshold = summary_threshold or self.CONFIG["summary_threshold"]
        
        # Enhanced context tracking
        self.important_contexts = {}
        self.topic_chains = []
        self.entity_frequencies = defaultdict(lambda: defaultdict(int))
        self.entity_relationships = defaultdict(set)  # Track entity co-occurrences
        self.topic_transitions = []  # Track how topics change
        self.intent_history = []  # Track conversation flow
        
        # User state
        self.last_clarification = None
        self.user_preferences = {}
        self.user_satisfaction = defaultdict(list)  # Track user satisfaction signals
        self.interaction_patterns = defaultdict(int)  # Track user interaction patterns
        
        # Medical context
        self.medical_context = {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "allergies": [],
            "previous_visits": []
        }
        
        # Temporal tracking
        self.session_start = datetime.now()
        self.last_activity = None
        self.turn_durations = []
    
    def add_turn(self, user_query, assistant_response, extracted_entities_map):
        """Add a conversation turn with enhanced context tracking"""
        turn_index = len(self.history)
        turn_start_time = datetime.now()
        
        # Track topic transitions
        current_topics = self._extract_topics(user_query)
        if self.topic_chains and any(topic in self.topic_chains[-1] for topic in current_topics):
            # Continuing previous topic
            self.topic_chains[-1].update(current_topics)
            self.topic_transitions.append(("continue", list(self.topic_chains[-1]), list(current_topics)))
        else:
            # New topic
            if self.topic_chains:
                self.topic_transitions.append(("switch", list(self.topic_chains[-1]), list(current_topics)))
            self.topic_chains.append(current_topics)
        
        # Analyze turn characteristics
        turn_analysis = self._analyze_turn(user_query, assistant_response)
        
        # Add the turn to history with enhanced metadata
        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "turn_index": turn_index,
            "timestamp": turn_start_time.isoformat(),
            "topics": list(current_topics),
            "entities": extracted_entities_map.copy() if extracted_entities_map else {},
            "analysis": turn_analysis,
            "duration": (datetime.now() - turn_start_time).total_seconds()
        })

        # Process entities with relationship tracking
        if extracted_entities_map:
            self._process_entities(extracted_entities_map, turn_index, turn_start_time)
            
        # Update temporal tracking
        self.last_activity = turn_start_time
        self.turn_durations.append((datetime.now() - turn_start_time).total_seconds())
        
        # Prune history if needed
        if len(self.history) > self.max_history_turns:
            self._prune_history()
            
        # Update interaction patterns
        self._update_interaction_patterns(user_query, assistant_response)
        
        # Generate turn summary for future reference
        turn_summary = self._generate_turn_summary(self.history[-1])
        self.history[-1]["summary"] = turn_summary

    def _process_entities(self, entities_map, turn_index, timestamp):
        """Process entities with enhanced relationship tracking and optimized performance."""
        all_entities = []
        for entity_type, entity_list in entities_map.items():
            if not isinstance(entity_list, list):
                entity_list = [entity_list]

            for entity_value in entity_list:
                if not entity_value:
                    continue

                all_entities.append((entity_type, entity_value))

                # Update entity frequency and last seen info
                self.entity_frequencies[entity_type][entity_value] += 1
                self.last_entity_by_type[entity_type] = {
                    "value": entity_value,
                    "turn_index": turn_index,
                    "timestamp": timestamp.isoformat(),
                    "frequency": self.entity_frequencies[entity_type][entity_value]
                }

                # Update medical context if relevant
                self._update_medical_context(entity_type, entity_value)

        # Update entity relationships in a single pass
        for i, (type1, value1) in enumerate(all_entities):
            # Add to contextual entities cache
            self.contextual_entities[f"{type1}:{value1}"] = {
                "value": value1,
                "type": type1,
                "turn_index": turn_index,
                "timestamp": timestamp.isoformat(),
                "frequency": self.entity_frequencies[type1][value1]
            }

            for type2, value2 in all_entities[i + 1:]:
                self.entity_relationships[f"{type1}:{value1}"].add(f"{type2}:{value2}")
                self.entity_relationships[f"{type2}:{value2}"].add(f"{type1}:{value1}")

    def _update_medical_context(self, entity_type, entity_value):
        """Update medical context based on entity type"""
        if entity_type == "symptom":
            self.medical_context["symptoms"].append({
                "value": entity_value,
                "timestamp": datetime.now().isoformat()
            })
        elif entity_type == "condition":
            self.medical_context["conditions"].append({
                "value": entity_value,
                "timestamp": datetime.now().isoformat()
            })
        elif entity_type == "medication":
            self.medical_context["medications"].append({
                "value": entity_value,
                "timestamp": datetime.now().isoformat()
            })
        elif entity_type == "allergy":
            self.medical_context["allergies"].append({
                "value": entity_value,
                "timestamp": datetime.now().isoformat()
            })

    def _analyze_turn(self, user_query, assistant_response):
        """Analyze turn characteristics"""
        return {
            "query_length": len(user_query.split()),
            "response_length": len(assistant_response.split()),
            "query_sentiment": self._analyze_sentiment(user_query),
            "contains_question": "?" in user_query,
            "contains_medical_terms": self._contains_medical_terms(user_query),
            "urgency_level": self._detect_urgency(user_query),
            "clarity_score": self._assess_clarity(user_query)
        }

    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = {"thank", "thanks", "good", "great", "excellent", "helpful", "perfect", "appreciate"}
        negative_words = {"bad", "poor", "unhelpful", "wrong", "incorrect", "not", "never", "terrible"}
        
        words = set(text.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _contains_medical_terms(self, text):
        """Check for medical terminology"""
        medical_indicators = {
            "pain", "symptom", "doctor", "medicine", "treatment",
            "diagnosis", "prescription", "condition", "emergency"
        }
        return bool(set(text.lower().split()).intersection(medical_indicators))

    def _detect_urgency(self, text):
        """Detect query urgency level"""
        urgent_indicators = {
            "emergency", "urgent", "immediately", "asap", "critical",
            "severe", "serious", "now", "hurry", "quick"
        }
        text_words = set(text.lower().split())
        urgency_count = len(text_words.intersection(urgent_indicators))
        
        if urgency_count >= 2:
            return "high"
        elif urgency_count == 1:
            return "medium"
        return "low"

    def _assess_clarity(self, text):
        """Assess query clarity"""
        clarity_score = 1.0
        
        # Check for ambiguous pronouns
        ambiguous_pronouns = {"it", "this", "that", "they", "them", "their", "these", "those"}
        if any(word in text.lower().split() for word in ambiguous_pronouns):
            clarity_score -= 0.2
            
        # Check query length
        if len(text.split()) < 3:
            clarity_score -= 0.3
        elif len(text.split()) > 15:
            clarity_score -= 0.1
            
        # Check for specific entities
        if any(char.isdigit() for char in text):  # Contains numbers (e.g., room numbers)
            clarity_score += 0.1
            
        return max(0.0, min(1.0, clarity_score))

    def _update_interaction_patterns(self, user_query, assistant_response):
        """Track user interaction patterns"""
        # Track query types
        if "?" in user_query:
            self.interaction_patterns["questions"] += 1
        if len(user_query.split()) < 5:
            self.interaction_patterns["short_queries"] += 1
        if self._contains_medical_terms(user_query):
            self.interaction_patterns["medical_queries"] += 1
            
        # Track satisfaction signals
        sentiment = self._analyze_sentiment(user_query)
        self.user_satisfaction[sentiment].append(datetime.now().isoformat())

    def _generate_turn_summary(self, turn):
        """Generate a concise summary of the turn"""
        return {
            "main_topic": next(iter(turn["topics"])) if turn["topics"] else None,
            "key_entities": list(turn["entities"].items())[:3] if turn["entities"] else [],
            "interaction_type": "question" if "?" in turn["user"] else "statement",
            "medical_context": self._contains_medical_terms(turn["user"]),
            "urgency": self._detect_urgency(turn["user"])
        }

    def get_conversation_analytics(self):
        """Get comprehensive conversation analytics"""
        if not self.history:
            return {}
            
        return {
            "session_metrics": {
                "total_turns": len(self.history),
                "session_duration": (datetime.now() - self.session_start).total_seconds(),
                "avg_turn_duration": sum(self.turn_durations) / len(self.turn_durations),
                "last_activity": self.last_activity.isoformat() if self.last_activity else None
            },
            "topic_analysis": {
                "unique_topics": len(set(topic for chain in self.topic_chains for topic in chain)),
                "topic_transitions": self.topic_transitions,
                "most_common_topics": Counter(topic for chain in self.topic_chains for topic in chain).most_common(3)
            },
            "entity_analysis": {
                "unique_entities": {
                    entity_type: len(entities)
                    for entity_type, entities in self.entity_frequencies.items()
                },
                "most_frequent_entities": {
                    entity_type: Counter(entities).most_common(3)
                    for entity_type, entities in self.entity_frequencies.items()
                },
                "entity_relationships": dict(self.entity_relationships)
            },
            "user_behavior": {
                "interaction_patterns": dict(self.interaction_patterns),
                "satisfaction_trends": {
                    sentiment: len(timestamps)
                    for sentiment, timestamps in self.user_satisfaction.items()
                },
                "clarity_trend": [
                    turn["analysis"]["clarity_score"]
                    for turn in self.history
                    if "analysis" in turn
                ]
            },
            "medical_context": {
                k: len(v) for k, v in self.medical_context.items()
            }
        }

    def get_smart_context(self, query: str, max_turns: int = None) -> dict:
        """Get relevant context based on query similarity, recency, and entity matching."""
        max_turns = max_turns or self.CONFIG["context_max_turns"]
        context = {
            "relevant_turns": [],
            "active_topics": set(),
            "important_entities": defaultdict(list),
            "relevance_scores": {},
            "medical_context": {},
            "user_preferences": {},
            "interaction_summary": {}
        }

        if not self.history:
            return context

        # Pre-calculate query topics and entities for efficiency
        query_topics = self._extract_topics(query)
        query_entities = self._extract_entities(query)

        # Score recent turns by relevance
        scored_turns = []
        for turn in self.history[-max_turns:]:
            score = self._calculate_turn_relevance(turn, query, query_topics, query_entities)
            scored_turns.append((turn, score))

        # Sort by relevance score
        scored_turns.sort(key=lambda x: x[1], reverse=True)

        # Add most relevant turns and their context
        for turn, score in scored_turns[:3]:  # Use top 3 most relevant
            if score > self.CONFIG["context_relevance_threshold"]:
                self._add_turn_to_context(turn, score, context)

        # Add other relevant context
        if self._contains_medical_terms(query):
            context["medical_context"] = {
                k: v[-3:] for k, v in self.medical_context.items() if v  # Last 3 entries
            }
        context["user_preferences"] = self.get_user_preferences()
        context["interaction_summary"] = self._get_interaction_summary()

        return context

    def _get_recent_satisfaction_level(self):
        """Calculate recent user satisfaction level"""
        if not self.user_satisfaction:
            return "neutral"
            
        recent_sentiments = []
        for sentiment, timestamps in self.user_satisfaction.items():
            recent_sentiments.extend([sentiment] * len(
                [ts for ts in timestamps if 
                 (datetime.now() - datetime.fromisoformat(ts)).total_seconds() < 300]  # Last 5 minutes
            ))
        
        if not recent_sentiments:
            return "neutral"
            
        sentiment_counts = Counter(recent_sentiments)
        return sentiment_counts.most_common(1)[0][0]

    def _determine_conversation_stage(self):
        """Determine the current stage of the conversation"""
        if not self.history:
            return "initial"
            
        num_turns = len(self.history)
        if num_turns <= 2:
            return "opening"
            
        # Check for closing signals in last turn
        last_turn = self.history[-1]
        if any(word in last_turn["user"].lower() for word in ["thank", "thanks", "bye", "goodbye"]):
            return "closing"
            
        # Check for problem resolution
        if num_turns > 5:
            recent_satisfaction = self._get_recent_satisfaction_level()
            if recent_satisfaction == "positive":
                return "resolution"
            elif recent_satisfaction == "negative":
                return "problem_solving"
                
        return "ongoing"

    def _extract_entities(self, text):
        """Simple entity extraction for context matching"""
        entities = set()
        # Add any capitalized words as potential entities
        entities.update(word for word in text.split() if word[0].isupper())
        # Add any numbers as potential entities
        entities.update(word for word in text.split() if any(c.isdigit() for c in word))
        return entities

    def get_last_entity_by_priority(self, type_priority=None):
        """Get most relevant entity based on priority, frequency, and recency"""
        if type_priority is None:
            type_priority = [
                "doctors", "departments", "services", "rooms", 
                "floors", "hospitals", "buildings"
            ]
            
        # Consider both priority and frequency/recency
        best_entity = None
        best_score = -1
        
        for entity_type in type_priority:
            if entity_type in self.last_entity_by_type:
                entity_data = self.last_entity_by_type[entity_type]
                # Score based on type priority, frequency, and recency
                type_weight = len(type_priority) - type_priority.index(entity_type)
                frequency_weight = entity_data["frequency"]
                recency_weight = entity_data["turn_index"] / len(self.history)
                
                score = (type_weight * 0.5) + (frequency_weight * 0.3) + (recency_weight * 0.2)
                
                if score > best_score:
                    best_score = score
                    best_entity = entity_data["value"]
        
        return best_entity

    def get_contextual_history_text(self, num_turns=5):
        """Get formatted conversation history with enhanced context"""
        if not self.history:
            return ""
            
        relevant_turns = self.history[-num_turns:]
        context_parts = []
        
        # Add active topic if any
        if self.current_topic:
            context_parts.append(f"Current topic: {self.current_topic['type']} - {self.current_topic['value']}")
        
        # Add conversation turns with entity highlights
        for turn in relevant_turns:
            # Format user query
            user_text = f"User: {turn['user']}"
            if turn.get("entities"):
                user_text += f" [Entities: {', '.join(f'{k}={v}' for k,v in turn['entities'].items())}]"
            context_parts.append(user_text)
            
            # Format assistant response
            if turn.get("assistant"):
                context_parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(context_parts)

    def get_relevant_entities_from_recent_turns(self, turns_to_check=3):
        """Get relevant entities from recent conversation turns"""
        if not self.history:
            return {}
            
        relevant_entities = defaultdict(list)
        recent_turns = self.history[-turns_to_check:]
        
        for turn in recent_turns:
            if turn.get("entities"):
                for entity_type, entities in turn["entities"].items():
                    if isinstance(entities, list):
                        relevant_entities[entity_type].extend(entities)
                    else:
                        relevant_entities[entity_type].append(entities)
        
        # Deduplicate while preserving order
        for entity_type in relevant_entities:
            relevant_entities[entity_type] = list(dict.fromkeys(relevant_entities[entity_type]))
        
        return dict(relevant_entities)

    def _extract_topics(self, text):
        """Extract topics from text using simple keyword matching"""
        topics = set()
        text_lower = text.lower()
        
        # Topic keywords (can be expanded)
        topic_patterns = {
            "location": ["where", "location", "find", "direction"],
            "appointment": ["book", "schedule", "appointment"],
            "information": ["what", "how", "explain", "tell"],
            "contact": ["contact", "phone", "email", "reach"],
            "emergency": ["emergency", "urgent", "immediate"],
            "facilities": ["room", "department", "facility"],
            "medical": ["treatment", "medicine", "procedure"]
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.add(topic)
        
        return topics

    def _prune_history(self):
        """Prune conversation history while preserving important context"""
        if len(self.history) <= self.max_history_turns:
            return
            
        # Keep first turn (initial context) and recent turns
        keep_count = self.max_history_turns - 1
        self.history = [self.history[0]] + self.history[-keep_count:]
        
        # Update turn indices
        for i, turn in enumerate(self.history):
            turn["turn_index"] = i
        
        # Update entity references
        self.contextual_entities = [
            entity for entity in self.contextual_entities
            if entity["turn_index"] == 0 or entity["turn_index"] >= len(self.history) - keep_count
        ]
        
        # Update topic chains
        if len(self.topic_chains) > self.max_history_turns:
            self.topic_chains = self.topic_chains[-self.max_history_turns:]

    def add_user_preference(self, preference_type, value):
        """Store user preferences for personalized responses"""
        self.user_preferences[preference_type] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

    def get_user_preferences(self):
        """Get user preferences for personalization"""
        return self.user_preferences

    def add_clarification(self, query, clarification):
        """Track when clarification was needed"""
        self.last_clarification = {
            "query": query,
            "clarification": clarification,
            "timestamp": datetime.now().isoformat()
        }

    def should_clarify(self, query):
        """Determine if clarification might be needed"""
        if not self.last_clarification:
            return False
            
        # Check if current query is similar to the one that needed clarification
        if semantic_similarity(query, self.last_clarification["query"]) > 0.8:
            return True
            
        return False
    
    def should_clarify_enhanced(self, query: str) -> tuple:
        """Enhanced clarification check with reason"""
        if not self.last_clarification:
            return False, None
            
        # Check various clarification triggers
        if len(query.split()) <= 3:
            return True, "short_query"
            
        if semantic_similarity(query, self.last_clarification["query"]) > 0.8:
            return True, "similar_to_last_clarification"
            
        # Check for ambiguous pronouns
        pronouns = {"it", "this", "that", "they", "them", "their", "these", "those"}
        if any(word in query.lower().split() for word in pronouns):
            return True, "ambiguous_pronouns"
        
        return False, None
    
    def get_followup_suggestions(self, current_intent: str, entities: dict) -> list:
        """Generate context-aware followup suggestions"""
        suggestions = []
        
        # Intent-based suggestions
        intent_suggestions = {
            "doctor_availability": [
                "Would you like to book an appointment?",
                "Would you like to see their schedule?",
                "Would you like to know their specialization?"
            ],
            "location": [
                "Would you like directions?",
                "Would you like to know the operating hours?",
                "Would you like to know about parking facilities?"
            ],
            "department_info": [
                "Would you like to know about the doctors?",
                "Would you like to know about services offered?",
                "Would you like to know the timings?"
            ]
        }
        
        if current_intent in intent_suggestions:
            suggestions.extend(intent_suggestions[current_intent])
        
        # Entity-based suggestions
        if entities.get("doctors"):
            suggestions.extend([
                f"Would you like to know more about Dr. {entities['doctors'][0]}?",
                "Would you like to see their availability?"
            ])
        elif entities.get("departments"):
            suggestions.extend([
                f"Would you like to know about doctors in {entities['departments'][0]}?",
                f"Would you like to know the location of {entities['departments'][0]}?"
            ])
        
        # Add general suggestions if no specific ones
        if not suggestions:
            suggestions.extend([
                "Would you like to know about our departments?",
                "Would you like to book an appointment?",
                "Would you like to know our location?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions


class RedisUserMemoryStore:
    def __init__(self, redis_url="redis://localhost:6379", use_summary=True):
        """Initialize with enhanced configuration"""
        self.redis_url = redis_url
        self.use_summary = use_summary
        self.lock = threading.Lock()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
        
        # Configuration
        self.summary_update_threshold = 5  # Number of turns before updating summary
        self.max_memory_age = timedelta(days=7)  # Maximum age of stored memories
        self.cache_ttl = 300  # Cache TTL in seconds
        self.batch_size = 100  # Batch size for bulk operations
        
        # Memory management
        self.memory_cache = TTLCache(maxsize=1000, ttl=self.cache_ttl)
        self.summary_cache = TTLCache(maxsize=500, ttl=self.cache_ttl * 2)
        self.context_cache = TTLCache(maxsize=500, ttl=self.cache_ttl)
        
        # Performance monitoring
        self.metrics = defaultdict(lambda: {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "latency": []
        })
        
        # Initialize Redis connection pool
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=10,
            decode_responses=True
        )

    def get(self, user_id):
        """Get user memory with enhanced caching and error handling"""
        start_time = time.time()
        
        try:
            # Check cache first
            if user_id in self.memory_cache:
                self.metrics["cache"]["hits"] += 1
                return self.memory_cache[user_id]
            
            self.metrics["cache"]["misses"] += 1
            
            with self.lock:
                memory = ConversationMemory()
                
                # Get Redis connection from pool
                redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                try:
                    # Load conversation history
                    memory = self._load_conversation_history(user_id, redis_client)
                    
                    # Load important context
                    context = self._get_important_context(user_id, redis_client)
                    if context:
                        memory.important_contexts.update(context)
                    
                    # Load user preferences
                    prefs = self._get_user_preferences(user_id, redis_client)
                    if prefs:
                        memory.user_preferences.update(prefs)
                    
                    # Cache the result
                    self.memory_cache[user_id] = memory
                    
                    # Update metrics
                    self.metrics["load"]["latency"].append(time.time() - start_time)
                    
                    return memory
                    
                except Exception as e:
                    self.metrics["load"]["errors"] += 1
                    logger.error(f"Error loading memory for user {user_id}: {e}")
                    return ConversationMemory()
                
        finally:
            # Cleanup and monitoring
            self._update_metrics("get", start_time)

    def _load_conversation_history(self, user_id, redis_client):
        """Load conversation history with optimized processing"""
        memory = ConversationMemory()
        
        try:
            # Get message history
            chat_history = RedisChatMessageHistory(redis_client=redis_client, session_id=user_id)
            
            if self.use_summary:
                # Use cached summary if available
                summary_key = f"summary:{user_id}"
                if summary_key in self.summary_cache:
                    memory.conversation_summary = self.summary_cache[summary_key]
                
                # Load and process messages
                messages = self._load_messages_with_summary(chat_history, memory)
            else:
                messages = self._load_messages_without_summary(chat_history)
            
            # Process messages in batches
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                self._process_message_batch(batch, memory)
            
            return memory
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return memory

    def _load_messages_with_summary(self, chat_history, memory):
        """Load messages with summarization support"""
        try:
            buffer_memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=chat_history,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=2048
            )
            
            messages = buffer_memory.chat_memory.messages
            
            # Update summary if needed
            if len(messages) >= self.summary_update_threshold:
                summary = self._generate_conversation_summary(
                    messages[-self.summary_update_threshold:]
                )
                if summary:
                    memory.conversation_summary = summary
                    self.summary_cache[f"summary:{chat_history.session_id}"] = summary
            
            return messages
            
        except Exception as e:
            logger.error(f"Error loading messages with summary: {e}")
            return []

    def _load_messages_without_summary(self, chat_history):
        """Load messages without summarization"""
        try:
            buffer_memory = ConversationBufferMemory(
                chat_memory=chat_history,
                memory_key="chat_history",
                return_messages=True
            )
            return buffer_memory.chat_memory.messages
        except Exception as e:
            logger.error(f"Error loading messages without summary: {e}")
            return []

    def _process_message_batch(self, messages, memory):
        """Process a batch of messages efficiently"""
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                try:
                    user_msg = messages[i]
                    ai_msg = messages[i + 1]
                    
                    # Extract metadata
                    entities = self._extract_entities_from_message(user_msg)
                    timestamp = user_msg.additional_kwargs.get("timestamp")
                    
                    # Add turn with enhanced metadata
                    memory.add_turn(
                        user_msg.content,
                        ai_msg.content,
                        entities,
                        metadata={
                            "timestamp": timestamp,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing message batch: {e}")
                    continue

    def save(self, user_id, memory: ConversationMemory):
        """Save memory with enhanced reliability and performance"""
        start_time = time.time()
        
        try:
            with self.lock:
                redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                # Save in batches
                self._save_conversation_history(user_id, memory, redis_client)
                self._save_important_context(user_id, memory.important_contexts, redis_client)
                self._save_user_preferences(user_id, memory.user_preferences, redis_client)
                
                # Update cache
                self.memory_cache[user_id] = memory
                
                # Cleanup old data
                self._cleanup_old_memories(user_id, redis_client)
                
        except Exception as e:
            self.metrics["save"]["errors"] += 1
            logger.error(f"Error saving memory for user {user_id}: {e}")
            
        finally:
            self._update_metrics("save", start_time)

    def _save_conversation_history(self, user_id, memory, redis_client):
        """Save conversation history efficiently"""
        try:
            chat_history = RedisChatMessageHistory(redis_client=redis_client, session_id=user_id)
            
            # Clear existing history
            chat_history.clear()
            
            # Prepare messages in batches
            messages = []
            for turn in memory.history:
                messages.extend([
                    HumanMessage(
                        content=turn["user"],
                        additional_kwargs={
                            "entities": turn.get("entities", {}),
                            "timestamp": turn.get("timestamp", datetime.now().isoformat()),
                            "metadata": turn.get("metadata", {})
                        }
                    ),
                    AIMessage(
                        content=turn["assistant"],
                        additional_kwargs={
                            "timestamp": turn.get("timestamp", datetime.now().isoformat())
                        }
                    )
                ])
            
            # Save in batches
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                for msg in batch:
                    chat_history.add_message(msg)
                
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
            raise

    def _save_important_context(self, user_id, context_data, redis_client):
        """Save important context with versioning"""
        try:
            context_key = f"important_context:{user_id}"
            
            # Add version and timestamp
            versioned_data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "data": context_data
            }
            
            # Save with expiration
            redis_client.set(
                context_key,
                json.dumps(versioned_data),
                ex=int(self.max_memory_age.total_seconds())
            )
            
            # Update cache
            self.context_cache[context_key] = versioned_data
            
        except Exception as e:
            logger.error(f"Error saving important context: {e}")
            raise

    def _save_user_preferences(self, user_id, preferences, redis_client):
        """Save user preferences"""
        try:
            prefs_key = f"user_preferences:{user_id}"
            
            # Add metadata
            enhanced_prefs = {
                "preferences": preferences,
                "updated_at": datetime.now().isoformat()
            }
            
            redis_client.set(
                prefs_key,
                json.dumps(enhanced_prefs),
                ex=int(self.max_memory_age.total_seconds())
            )
            
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            raise

    def _get_user_preferences(self, user_id, redis_client):
        """Get user preferences"""
        try:
            prefs_key = f"user_preferences:{user_id}"
            prefs_data = redis_client.get(prefs_key)
            
            if prefs_data:
                prefs = json.loads(prefs_data)
                return prefs.get("preferences", {})
            return {}
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}

    def _generate_conversation_summary(self, messages):
        """Generate enhanced conversation summary"""
        try:
            # Convert messages to conversation format
            conversation = "\n".join([
                f"User: {msg.content if isinstance(msg, HumanMessage) else ''}"
                f"Assistant: {msg.content if isinstance(msg, AIMessage) else ''}"
                for msg in messages
            ])
            
            # Enhanced summary prompt
            summary_prompt = f"""Analyze and summarize this conversation, focusing on:
            1. Key Topics: Main subjects discussed
            2. User Intent: Primary goals or needs
            3. Important Information: Critical details or decisions
            4. Context: Relevant background information
            5. Next Steps: Any pending actions or follow-ups
            6. User Preferences: Any stated preferences or requirements
            7. Medical Context: Any medical information or concerns mentioned
            
            Conversation:
            {conversation}
            
            Provide a structured summary covering these aspects:"""
            
            response = self.llm.invoke(summary_prompt)
            
            return {
                "content": response.content,
                "generated_at": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    def clear(self, user_id):
        """Clear user memory with enhanced cleanup"""
        start_time = time.time()
        
        try:
            with self.lock:
                redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                # Clear all user-related data
                patterns = [
                    f"message_store:{user_id}",
                    f"important_context:{user_id}",
                    f"user_preferences:{user_id}",
                    f"summary:{user_id}"
                ]
                
                for pattern in patterns:
                    redis_client.delete(pattern)
                
                # Clear caches
                if user_id in self.memory_cache:
                    del self.memory_cache[user_id]
                if f"summary:{user_id}" in self.summary_cache:
                    del self.summary_cache[f"summary:{user_id}"]
                if f"important_context:{user_id}" in self.context_cache:
                    del self.context_cache[f"important_context:{user_id}"]
                
        except Exception as e:
            self.metrics["clear"]["errors"] += 1
            logger.error(f"Error clearing memory for user {user_id}: {e}")
            
        finally:
            self._update_metrics("clear", start_time)

    def _cleanup_old_memories(self, user_id, redis_client):
        """Enhanced memory cleanup with analytics"""
        try:
            # Get all user-related keys
            user_keys = redis_client.keys(f"*:{user_id}")
            current_time = datetime.now()
            cleaned_count = 0
            
            for key in user_keys:
                try:
                    # Get timestamp from data
                    data = redis_client.get(key)
                    if data:
                        data_dict = json.loads(data)
                        timestamp = data_dict.get("timestamp") or data_dict.get("updated_at")
                        
                        if timestamp:
                            stored_time = datetime.fromisoformat(timestamp)
                            if (current_time - stored_time) > self.max_memory_age:
                                redis_client.delete(key)
                                cleaned_count += 1
                                
                except json.JSONDecodeError:
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

    def _update_metrics(self, operation, start_time):
        """Update performance metrics"""
        duration = time.time() - start_time
        self.metrics[operation]["latency"].append(duration)
        
        # Keep only recent latency measurements
        if len(self.metrics[operation]["latency"]) > 1000:
            self.metrics[operation]["latency"] = self.metrics[operation]["latency"][-1000:]

    def get_metrics(self):
        """Get performance metrics"""
        metrics_summary = {}
        
        for operation, data in self.metrics.items():
            latencies = data["latency"]
            metrics_summary[operation] = {
                "hits": data["hits"],
                "misses": data["misses"],
                "errors": data["errors"],
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            }
        
        return metrics_summary

    def get_memory_stats(self):
        """Get memory usage statistics"""
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        try:
            info = redis_client.info()
            return {
                "used_memory": info["used_memory_human"],
                "peak_memory": info["used_memory_peak_human"],
                "total_keys": info["db0"]["keys"],
                "connected_clients": info["connected_clients"],
                "total_connections_received": info["total_connections_received"],
                "total_commands_processed": info["total_commands_processed"]
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

load_dotenv()

# --- Configuration for Hospital Data ---
FAISS_INDEX_PATH = "rag_aimms_jammu"
HOSPITAL_MODEL_JSON_PATH = "hospital_building.json"
QA_PAIRS_JSON_PATH = "jammu_qa_pairs_cleaned.json"
CHAT_MEMORY_INDEX_PATH = "chat_memory_index"
os.makedirs(CHAT_MEMORY_INDEX_PATH, exist_ok=True)
ENTITY_GROUNDING_THRESHOLD = float(os.getenv("ENTITY_GROUNDING_THRESHOLD", 0.65))
USE_LLM_INTENT_FALLBACK = os.getenv("USE_LLM_INTENT_FALLBACK", "true").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALLOWED_FILES = {os.path.basename(HOSPITAL_MODEL_JSON_PATH),
                 os.path.basename(QA_PAIRS_JSON_PATH)}

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY not found in environment variables.")

chat_memory_faiss = None
chat_memory_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def initialize_chat_memory_faiss():
    global chat_memory_faiss
    try:
        chat_memory_faiss = FAISS.load_local(
            CHAT_MEMORY_INDEX_PATH,
            chat_memory_embedding,
            allow_dangerous_deserialization=True
        )
        logger.info("Loaded chat memory FAISS index.")
    except Exception as e:
        logger.warning(f"Creating new chat memory FAISS index: {e}")

        from langchain_core.documents import Document
        dummy_doc = Document(page_content="placeholder", metadata={"init": True})

        try:
            chat_memory_faiss = FAISS.from_documents([dummy_doc], chat_memory_embedding)
            chat_memory_faiss.save_local(CHAT_MEMORY_INDEX_PATH)
            logger.info("Initialized new empty FAISS index for chat memory.")
        except Exception as inner_e:
            logger.error(f"Failed to initialize FAISS index: {inner_e}")
            raise

initialize_chat_memory_faiss()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We can replace "*" with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom in-memory user memory store
user_memory_store = RedisUserMemoryStore(use_summary=True)

# === UTILITY FUNCTIONS ===
def ensure_list(val):
    """Ensure a value is returned as a list"""
    if isinstance(val, list): 
        return val
    elif val: 
        return [val]
    return []


def format_operating_hours(hours_data):
    """Format operating hours data into readable string"""
    if not hours_data: 
        return "N/A"

    order = ["mondayToFriday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
    parts = []
    
    for day in order:
        if day not in hours_data: 
            continue
        
        # Format day label
        if day == "mondayToFriday":
            label = "Mon–Fri"
        else:
            label = day.capitalize()
        
        time = hours_data[day]
        
        if isinstance(time, dict):
            start_time = time.get('start', 'N/A')
            end_time = time.get('end', 'N/A')
            parts.append(f"{label}: {start_time}–{end_time}")
        else:
            parts.append(f"{label}: {str(time)}")
    
    return "; ".join(parts) if parts else "N/A"

def format_response_channels(channels_list):
    """Format response channels data into readable string"""
    if not channels_list:
        return "N/A"

    parts = []
    
    for ch in channels_list:
        ch_type = ch.get('type', 'Unknown')
        description = ch.get('description', '')
        contact = ch.get('contact', {})
        
        # Collect contact information
        contact_parts = []
        if contact.get('phone'):
            contact_parts.append(f"Phone: {contact['phone']}")
        if contact.get('email'):
            contact_parts.append(f"Email: {contact['email']}")
        if contact.get('website'):
            contact_parts.append(f"Profile website: {contact['website']}")

        contact_str = ", ".join(contact_parts) if contact_parts else "N/A"
        
        # Build channel information
        channel_info = f"{ch_type}"
        if description:
            channel_info += f" ({description})"
        channel_info += f": {contact_str}"

        # Add operating hours if available
        op_hours = format_operating_hours(ch.get('operatingHours', {}))
        if op_hours != "N/A":
            channel_info += f" [Hours: {op_hours}]"

        parts.append(channel_info)

    return ". ".join(parts)


def format_doctor_availability(availability_data):
    """Format doctor availability data into readable string"""
    if not availability_data:
        return "Not specified"
    
    days = ensure_list(availability_data.get("days", []))
    time = availability_data.get("time", "Not specified")
    
    if not days:
        return f"Time: {time}" if time != "Not specified" else "Availability: Not specified"
    
    return f"Days: {', '.join(days)}; Time: {time}"


def format_contact_info(contact_data):
    """Format contact information into readable string"""
    if not contact_data:
        return "N/A"

    parts = []
    
    if contact_data.get('phoneNumbers'):
        phone_nums = ensure_list(contact_data['phoneNumbers'])
        parts.append(f"Phone: {', '.join(phone_nums)}")
    
    if contact_data.get('email'):
        parts.append(f"Email: {contact_data['email']}")
    
    if contact_data.get('website'):
        parts.append(f"Website: {contact_data['website']}")
    
    address = contact_data.get('address', {})
    if address:
        addr_parts = []
        for field in ['street', 'city', 'state', 'postalCode', 'country']:
            if address.get(field):
                addr_parts.append(str(address[field]))
        if addr_parts:
            parts.append(f"Address: {', '.join(addr_parts)}")

    return "; ".join(parts) if parts else "N/A"


# === HOSPITAL DATA LOADER ===
class HospitalDataLoader:
    def __init__(self, hospital_filepath=HOSPITAL_MODEL_JSON_PATH, qa_filepath=QA_PAIRS_JSON_PATH):
        """Initialize with enhanced data loading and validation"""
        self.hospital_filepath = hospital_filepath
        self.qa_filepath = qa_filepath
        self.hospital_data = self.load_json_secure(self.hospital_filepath)
        self.qa_data = self.load_json_secure(self.qa_filepath)
        self.all_known_entities = self._extract_known_entities()

    def load_json_secure(self, filename):
        if not filename.endswith(".json") or os.path.basename(filename) not in ALLOWED_FILES:
            logger.error(f"Unauthorized or invalid file access attempt: {filename}")
            return None
        try:
            with open(filename, "r", encoding="utf-8") as file:
                data = json.load(file)
                logger.info(f"Successfully loaded {filename}")
                return data
        except FileNotFoundError:
            logger.error(f"Error: The file {filename} was not found.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {filename}: {e}")
            return None

    def _extract_known_entities(self):
        entities = {
            "rooms": [], "departments": [], "doctors": [], "services": [],
            "lifts": [], "stairs": [], "emergencyExits": [], "entrances": [], "ramps": [],
            "hospitals": [], "buildings": [], "venues": [],
            "qa_topics": [], "qa_sources": []
        }
        
        # Extract from hospital data
        if self.hospital_data:
            rooms, departments, doctors, services = set(), set(), set(), set()
            lifts, stairs, exits, entrances, ramps = set(), set(), set(), set(), set()
            hospitals, buildings, venues = set(), set(), set()

            for item in self.hospital_data:
                # Hospital information
                if item.get("hospitalName"):
                    hospitals.add(item["hospitalName"])

                # Location context
                loc_ctx = item.get("locationContext", {})
                if loc_ctx.get("venueName"):
                    venues.add(loc_ctx["venueName"])
                if loc_ctx.get("buildingName"):
                    buildings.add(loc_ctx["buildingName"])

                # Room details
                room_details = item.get("roomDetails", {})
                if room_details.get("roomName"):
                    rooms.add(room_details["roomName"])
                if room_details.get("roomNumber"):
                    rooms.add(str(room_details["roomNumber"]))

                # Departments and doctors
                for dept in ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        departments.add(dept["departmentName"])
                    
                    # Services from departments
                    for service_name in ensure_list(dept.get("relatedServices", [])):
                        services.add(service_name)
                    
                    # Doctors from departments
                    for doctor in ensure_list(dept.get("doctors", [])):
                        if doctor.get("name"):
                            doctors.add(doctor["name"])

                # Services offered at hospital level
                for service_item in ensure_list(item.get("servicesOffered", [])):
                    if service_item.get("serviceName"):
                        services.add(service_item["serviceName"])

                # Accessibility access points
                access = item.get("accessibility", {}).get("nearestAccessPoints", {})
                for group, collector in [("lifts", lifts), ("stairs", stairs), ("emergencyExits", exits),
                                        ("entrances", entrances), ("ramps", ramps)]:
                    for ap in ensure_list(access.get(group, [])):
                        if ap.get("name"):
                            collector.add(ap["name"])

            entities.update({
                "rooms": sorted(list(rooms)),
                "departments": sorted(list(departments)),
                "doctors": sorted(list(doctors)),
                "services": sorted(list(services)),
                "lifts": sorted(list(lifts)),
                "stairs": sorted(list(stairs)),
                "emergencyExits": sorted(list(exits)),
                "entrances": sorted(list(entrances)),
                "ramps": sorted(list(ramps)),
                "hospitals": sorted(list(hospitals)),
                "buildings": sorted(list(buildings)),
                "venues": sorted(list(venues))
            })
        
        # Extract from QA data
        if self.qa_data:
            topics = set()
            sources = set()
            
            for qa_item in self.qa_data:
                # Extract topics from context
                context = qa_item.get("context", "")
                if context:
                    topics.add(context)
                
                # Extract sources
                source = qa_item.get("source", "")
                if source:
                    sources.add(source)
            
            entities.update({
                "qa_topics": sorted(list(topics)),
                "qa_sources": sorted(list(sources))
            })

        return entities
    
    def get_all_metadata_tags(self):
        tags = set()
        
        # From hospital data - synthesize tags from various fields
        if self.hospital_data:
            for item in self.hospital_data:
                # Hospital type as tag
                if item.get("hospitalType"):
                    tags.add(item["hospitalType"].lower().replace(" ", "_"))
                
                # Room types as tags
                room_details = item.get("roomDetails", {})
                if room_details.get("roomType"):
                    tags.add(room_details["roomType"].lower().replace(" ", "_"))
                if room_details.get("roomSubType"):
                    tags.add(room_details["roomSubType"].lower().replace(" ", "_"))
                
                # Department names as tags
                for dept in ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        tags.add(dept["departmentName"].lower().replace(" ", "_"))
                
                # Services as tags
                for service in ensure_list(item.get("servicesOffered", [])):
                    if service.get("serviceName"):
                        tags.add(service["serviceName"].lower().replace(" ", "_"))
                
                # Emergency services tag
                if item.get("emergencyServices"):
                    tags.add("emergency_services")
                
                # Accessibility tags
                if item.get("accessibility", {}).get("isWheelchairAccessible"):
                    tags.add("wheelchair_accessible")
        
        # From QA data - extract implicit tags from context and questions
        if self.qa_data:
            for qa_item in self.qa_data:
                context = qa_item.get("context", "")
                if context:
                    context_tag = context.lower().replace(" ", "_")
                    tags.add(context_tag)
        
        return sorted(tags)

    def get_metadata_tag_counts(self):
        counter = Counter()
        
        # From hospital data
        if self.hospital_data:
            for item in self.hospital_data:
                # Count hospital types
                if item.get("hospitalType"):
                    counter[item["hospitalType"].lower().replace(" ", "_")] += 1
                
                # Count room types
                room_details = item.get("roomDetails", {})
                if room_details.get("roomType"):
                    counter[room_details["roomType"].lower().replace(" ", "_")] += 1
                
                # Count departments
                for dept in ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        counter[dept["departmentName"].lower().replace(" ", "_")] += 1
                
                # Count services
                for service in ensure_list(item.get("servicesOffered", [])):
                    if service.get("serviceName"):
                        counter[service["serviceName"].lower().replace(" ", "_")] += 1
        
        # From QA data
        if self.qa_data:
            for qa_item in self.qa_data:
                context = qa_item.get("context", "")
                if context:
                    context_tag = context.lower().replace(" ", "_")
                    counter[context_tag] += 1
        
        return dict(counter.most_common())


def prepare_documents():
    """Enhanced document preparation that combines hospital data and QA pairs"""
    if not data_loader.hospital_data and not data_loader.qa_data:
        logger.error("No data loaded. Cannot prepare documents.")
        return []

    documents = []

    # Process Hospital Data
    if data_loader.hospital_data:
        documents.extend(_prepare_hospital_documents())
    
    # Process QA Data
    if data_loader.qa_data:
        documents.extend(_prepare_qa_documents())
    
    logger.info(f"Prepared {len(documents)} total documents for FAISS index.")
    return documents


def _prepare_hospital_documents():
    """Prepare documents from hospital data (doctor rooms, accessibility, services, etc.)"""
    documents = []

    for item_index, item_data in enumerate(data_loader.hospital_data):
        content_parts = []
        metadata_payload = {
            "source_doc_id": item_data.get("id", f"hospital_item_{item_index}"),
            "document_type": "hospital_data",
            "type": item_data.get("physical", {}).get("type", "UnknownType").lower()
        }

        # Location Context
        loc = item_data.get("locationContext", {})
        content_parts.append(
            f"Location: Hospital '{loc.get('hospitalName', 'N/A')}', Type '{loc.get('hospitalType', 'N/A')}', "
            f"Bldg '{loc.get('buildingName', 'N/A')}', Tower '{loc.get('tower', 'N/A')}', "
            f"Flr {loc.get('floor', 'N/A')}, Zone '{loc.get('zone', 'N/A')}', {loc.get('areaType', 'N/A')}."
        )
        metadata_payload.update({
            "hospitalName": loc.get("hospitalName"),
            "hospitalType": loc.get("hospitalType"),
            "buildingName": loc.get("buildingName"),
            "tower": loc.get("tower"),
            "floor": str(loc.get("floor", "")),
            "zone": loc.get("zone"),
            "areaType": loc.get("areaType")
        })

        # Physical
        physical = item_data.get("physical", {})
        name = physical.get("name", f"Unnamed {metadata_payload['type']}")
        content_parts.append(
            f"Name: {name} (Type: {physical.get('type', 'N/A')}, SubType: {physical.get('subType', 'N/A')})."
        )
        metadata_payload.update({
            "room_name": name,
            "room_subtype": physical.get("subType")
        })

        struct = physical.get("structure", {})
        content_parts.append(
            f"Structure: Cap '{struct.get('capacity', 'N/A')}', Area {struct.get('areaSqFt', 'N/A')}sqft, "
            f"Shape '{struct.get('shape', 'N/A')}', Floor '{struct.get('flooringType', 'N/A')}'."
        )

        coords = physical.get("coordinates", {}).get("cartesian", {})
        door_coords = coords.get("door", {})
        content_parts.append(
            f"Coords: X={coords.get('x')}, Y={coords.get('y')}. "
            f"Door: X={door_coords.get('x')}, Y={door_coords.get('y')}."
        )
        metadata_payload["coordinates_door_cartesian"] = door_coords

        door = physical.get("door", {})
        content_parts.append(
            f"Door: Type '{door.get('type', 'N/A')}', Mech '{door.get('mechanism', 'N/A')}', "
            f"Motion '{door.get('motion', 'N/A')}'. SmartLock: {door.get('smartLock', False)}."
        )

        utilities = physical.get("utilities", {})
        utils_str = ", ".join(f"{k}: {v}" for k, v in utilities.items()) if utilities else "N/A"
        content_parts.append(f"Utilities: {utils_str}.")

        # Functional
        func = item_data.get("functional", {})
        content_parts.append(f"Purpose: {func.get('purpose', 'N/A')}. Access: {func.get('accessLevel', 'N/A')}.")
        metadata_payload["purpose"] = func.get("purpose")

        entity = func.get("associatedEntity", {})
        if entity.get("name"):
            content_parts.append(
                f"Entity: {entity.get('name')} ({entity.get('entityType', 'N/A')}). "
                f"Desc: {entity.get('about', 'N/A')[:100]}... "
                f"Ind: {entity.get('industry', 'N/A')}."
            )
            contact = entity.get("contact", {})
            contact_str = (
                f"Email: {contact.get('email', 'N/A')}, Phone: {contact.get('phone', 'N/A')}, "
                f"Profile Website: {contact.get('website', 'N/A')}."
            )
            content_parts.append(f"Entity Contact: {contact_str}")
            metadata_payload.update({
                "doctor_name": entity.get("name"),
                "doctor_contact_email": contact.get("email"),
                "doctor_contact_phone": contact.get("phone"),
                "doctor_website": contact.get("website"),
                "specialization": entity.get("specialization"),
                "industry": entity.get("industry", entity.get("specialization")),
                "department_name": entity.get("departmentName")
            })

            # Entity Operating Hours
            entity_hours = format_operating_hours(entity.get("operatingHours", {}))
            content_parts.append(f"Entity Hours: {entity_hours}.")

            services = ", ".join(ensure_list(entity.get("servicesOffered", []))) or "N/A"
            content_parts.append(f"Services: {services}.")

        # Availability
        avail = func.get("availability", {})
        avail_str = (
            f"Open: {', '.join(ensure_list(avail.get('daysOpen', [])))}. "
            f"Hours: {avail.get('startTime', 'N/A')} - {avail.get('endTime', 'N/A')}."
        ) if avail else "N/A"
        content_parts.append(f"Availability: {avail_str}")

        # Booking
        booking = func.get("booking", {})
        if booking.get("enabled"):
            book_str = (
                f"Method: {booking.get('method', 'N/A')}, URL: {booking.get('url', 'N/A')}. "
                f"Approval: {booking.get('approvalRequired', False)}. "
                f"Notes: {booking.get('notes', 'N/A')[:100]}..."
            )
            content_parts.append(f"Booking: {book_str}")

        # Response Channels
        response = format_response_channels(func.get("responseChannels", []))
        content_parts.append(f"Response Channels: {response}.")

        # Accessibility
        acc = item_data.get("accessibility", {})
        content_parts.append(f"Wheelchair: {acc.get('isWheelchairAccessible', False)}.")
        acc_features = ", ".join(ensure_list(acc.get("features", []))) or "N/A"
        content_parts.append(f"Access Features: {acc_features}.")
        metadata_payload["accessibility_features_summary"] = acc_features[:100]

        ap = acc.get("nearestAccessPoints", {})
        ap_parts = []
        for ap_type, ap_list in ap.items():
            for ap_item in ensure_list(ap_list):
                ap_parts.append(
                    f"{ap_item.get('name', f'Unnamed {ap_type[:-1]}')} "
                    f"({ap_item.get('distanceMeters', 'N/A')}m)"
                )
        content_parts.append(f"Nearest APs: {', '.join(ap_parts) if ap_parts else 'N/A'}.")

        # Amenities
        amenities = ", ".join(ensure_list(acc.get("amenities", []))) or "N/A"
        content_parts.append(f"Amenities: {amenities}.")
        metadata_payload["amenities_summary"] = amenities[:100]

        # Media images
        images = item_data.get("media", {}).get("images", [])
        image_urls = ", ".join(img.get("url") for img in images if img.get("url"))
        content_parts.append(f"Images: {image_urls or 'N/A'}.")

        # Status
        status = item_data.get("status", {})
        content_parts.append(
            f"Status: {'Op' if status.get('operational') else 'NonOp'}. "
            f"Maint: {status.get('underMaintenance', False)}. "
            f"Insp: {status.get('lastInspected', 'N/A')}."
        )

        # Metadata
        meta = item_data.get("metadata", {})
        tags = ", ".join(ensure_list(meta.get("tags", []))) or "N/A"
        content_parts.append(f"Tags: {tags}.")
        metadata_payload["tags"] = ensure_list(meta.get("tags", []))[:5]

        summary = meta.get("summary", "No summary.")[:200]
        content_parts.append(f"Summary: {summary}...")
        metadata_payload["summary"] = summary
        metadata_payload["priority"] = meta.get("priority", 1)

        # Final Document
        page_content = "\n".join(filter(None, content_parts))
        documents.append(Document(page_content=page_content, metadata=metadata_payload))

    logger.info(f"Prepared {len(documents)} hospital documents.")
    return documents


def _prepare_qa_documents():
    """Prepare documents from QA pairs data"""
    documents = []
    
    for qa_index, qa_item in enumerate(data_loader.qa_data):
        content_parts = []
        metadata_payload = {
            "source_doc_id": f"qa_item_{qa_index}",
            "document_type": "qa_data",
            "type": "knowledge_base"
        }
        
        # Extract basic information
        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        context = qa_item.get("context", "")
        source = qa_item.get("source", "")
        
        # Build content with clear structure
        if question:
            content_parts.append(f"Question: {question}")
            metadata_payload["question"] = question
        
        if answer:
            content_parts.append(f"Answer: {answer}")
            metadata_payload["answer"] = answer[:200]  # Store truncated version
        
        if context:
            content_parts.append(f"Context: {context}")
            metadata_payload["context"] = context
            metadata_payload["topic"] = context.lower().replace(" ", "_")
        
        if source:
            content_parts.append(f"Source: {source}")
            metadata_payload["source_url"] = source
        
        # Create searchable content combining question, answer, and context
        searchable_content = f"{question} {answer} {context}".strip()
        content_parts.append(f"Searchable Content: {searchable_content}")
        
        # Extract key terms from question and answer for better matching
        key_terms = set()
        for text in [question, answer, context]:
            if text:
                # Simple keyword extraction
                words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
                key_terms.update(words)
        
        metadata_payload["key_terms"] = list(key_terms)[:10]  # Limit to top 10
        
        # Categorize based on content
        content_lower = searchable_content.lower()
        if any(term in content_lower for term in ['aiims', 'hospital', 'medical', 'healthcare']):
            metadata_payload["category"] = "hospital_info"
        elif any(term in content_lower for term in ['doctor', 'physician', 'specialist']):
            metadata_payload["category"] = "medical_staff"
        elif any(term in content_lower for term in ['appointment', 'booking', 'schedule', 'helpdesk']):
            metadata_payload["category"] = "appointments"
        elif any(term in content_lower for term in ['department', 'ward', 'unit']):
            metadata_payload["category"] = "departments"
        elif any(term in content_lower for term in ['emergency', 'urgent', 'critical', 'helpdesk']):
            metadata_payload["category"] = "emergency"
        elif any(term in content_lower for term in ['location', 'address', 'where']):
            metadata_payload["category"] = "location"
        elif any(term in content_lower for term in ['contact', 'phone', 'email']):
            metadata_payload["category"] = "contact"
        elif any(term in content_lower for term in ['hours', 'time', 'open', 'close']):
            metadata_payload["category"] = "schedule"
        else:
            metadata_payload["category"] = "general"
        
        # Set priority based on question type
        if question.lower().startswith(('what is', 'what are')):
            metadata_payload["priority"] = 1  # High priority for definition questions
        elif question.lower().startswith(('how to', 'how can', 'how do')):
            metadata_payload["priority"] = 2  # Medium priority for how-to questions
        elif question.lower().startswith(('where is', 'where are')):
            metadata_payload["priority"] = 1  # High priority for location questions
        else:
            metadata_payload["priority"] = 3  # Lower priority for other questions
        
        # Create tags based on content
        tags = []
        if context:
            tags.append(context.lower().replace(" ", "_"))
        tags.append(metadata_payload["category"])
        metadata_payload["tags"] = tags[:3]
        
        # Create summary
        summary = f"Q&A about {context if context else 'hospital information'}: {question[:50]}..."
        metadata_payload["summary"] = summary

        # Final Document
        page_content = "\n".join(filter(None, content_parts))
        documents.append(Document(page_content=page_content, metadata=metadata_payload))
    
    logger.info(f"Prepared {len(documents)} QA documents.")
    return documents

data_loader = HospitalDataLoader()


# Enhanced NLU Processor to handle Hospital and QA content
class EnhancedNLUProcessor(NLUProcessor):
    def __init__(self):
        super().__init__()
        self.hospital_specific_intents = [
            # Clinical Intents
            "find_doctor", "find_specialist", "book_appointment",
            "check_availability", "get_medical_report", "check_test_results",
            "request_prescription", "medical_emergency", "get_treatment_info",
            
            # Administrative Intents
            "admission_process", "discharge_process", "payment_inquiry",
            "insurance_verification", "document_request", "feedback_submission",
            
            # Facility Intents
            "find_department", "locate_facility", "check_equipment",
            "parking_info", "cafeteria_info", "visiting_hours",
            
            # Service Intents
            "lab_services", "radiology_services", "pharmacy_services",
            "emergency_services", "ambulance_services", "telemedicine_services"
        ]
        
        self._init_hospital_patterns()
        self._init_department_mapping()
        self._init_medical_knowledge()
    
    def _init_hospital_patterns(self):
        """Initialize hospital-specific patterns"""
        self.hospital_patterns = {
            # Clinical Patterns
            "symptoms": r"\b(?:suffering from|experiencing|having|got|feel(?:ing)?)\s+([a-z\s,]+(?:pain|ache|discomfort|fever|cough|cold|nausea|vomiting|bleeding|swelling))",
            "medications": r"\b(?:medicine|tablet|capsule|syrup|injection|dose|medication)\s+([A-Za-z0-9\s-]+(?:mg|ml|mcg|g)?)",
            "vitals": r"\b(?:BP|blood pressure|pulse|temperature|SpO2|oxygen|heart rate|respiratory rate)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            
            # Test Patterns
            "lab_tests": r"\b(?:blood test|urine test|culture|biopsy|analysis|CBC|ESR|CRP|thyroid|sugar|creatinine)\b",
            "imaging": r"\b(?:x-ray|ultrasound|CT scan|MRI|ECG|EEG|PET scan|radiograph|sonography)\b",
            
            # Administrative Patterns
            "insurance": r"\b(?:insurance|policy|TPA|cashless|reimbursement)\s+([A-Za-z0-9\s-]+)",
            "billing": r"\b(?:bill|invoice|payment|cost|charge|fee|estimate|package)\s+([A-Za-z0-9\s-]+)",
            
            # Temporal Patterns
            "duration": r"\b(?:for|since|last|past)\s+(\d+)\s+(?:day|week|month|year)s?\b",
            "frequency": r"\b(\d+)\s+times?\s+(?:daily|weekly|monthly|yearly|a day|a week|a month|a year)\b"
        }
    
    def _init_medical_knowledge(self):
        """Initialize medical knowledge base"""
        self.medical_knowledge = {
            # Symptom Categories
            "pain_symptoms": ["headache", "backache", "chest pain", "abdominal pain", "joint pain"],
            "respiratory_symptoms": ["cough", "breathlessness", "wheezing", "chest congestion"],
            "digestive_symptoms": ["nausea", "vomiting", "diarrhea", "constipation", "indigestion"],
            "cardiac_symptoms": ["palpitations", "chest pain", "dizziness", "fainting"],
            
            # Disease Categories
            "chronic_conditions": ["diabetes", "hypertension", "asthma", "arthritis"],
            "acute_conditions": ["fever", "infection", "injury", "allergy"],
            "emergency_conditions": ["heart attack", "stroke", "severe bleeding", "fracture"],
            
            # Treatment Types
            "medication_types": ["antibiotics", "painkillers", "antivirals", "insulin"],
            "procedure_types": ["surgery", "physiotherapy", "chemotherapy", "radiation"],
            
            # Specialties (same names as departments)
            "specialties": self.department_mapping.copy()
        }
    
    def _init_department_mapping(self):
        """Initialize department mappings"""
        self.department_mapping = {
            # Clinical Departments
            "cardiology": ["heart", "cardiac", "chest", "cardio"],
            "neurology": ["brain", "nerve", "spine", "neuro"],
            "neurosurgery": ["neuro surgery", "brain surgery", "neurosurgical", "ns"],
            "orthopedics": ["bone", "joint", "fracture", "ortho", "orthopaedics"],
            "pediatrics": ["child", "infant", "newborn", "paediatrics", "child care", "pediat"],
            "gynecology": ["women", "pregnancy", "obstetrics", "obg", "obgy", "gynae", "labour room"],
            "dermatology": ["skin", "hair", "nail", "derma", "dermatologist"],
            "ophthalmology": ["eye", "vision", "retina", "ophth", "eye doctor"],
            "ent": ["ear", "nose", "throat", "ent specialist"],
            "psychiatry": ["mental", "depression", "anxiety", "psychology", "counseling"],
            "gastroenterology": ["stomach", "liver", "ulcer", "gastro", "digestive", "abdomen"],
            "medical_gastroenterology": ["med gastro", "medical gastro", "digestive medicine"],
            "urology": ["urine", "bladder", "prostate", "urology", "uro", "urinary", "kidney doctor"],
            "nephrology": ["kidney", "dialysis", "renal", "nephro"],
            "pulmonology": ["lungs", "breathing", "asthma", "respiratory", "tb"],
            "oncology": ["cancer", "tumor", "chemo", "radiotherapy", "onco"],
            "anesthesiology": ["anesthesia", "sedation", "pre-op", "anaesthetics"],
            "endocrinology": ["hormone", "thyroid", "diabetes", "endo", "insulin"],
            "general_medicine": ["internal medicine", "medicine", "general opd", "physician"],
            "general_surgery": ["gs", "surgical", "operation", "surgery department"],
            "pmr": ["pmr", "rehabilitation", "physiatry", "physical medicine", "physical rehab"],
            "palliative_care": ["palliative", "pain management", "terminal care"],

            # Support / Diagnostic Departments
            "radiology": ["x-ray", "scan", "mri", "ct", "ultrasound", "imaging"],
            "pathology": ["lab", "blood test", "sample", "histopathology"],
            "biochemistry": ["glucose", "lipid", "creatinine", "enzyme", "blood sugar"],
            "microbiology": ["infection", "culture test", "bacteria", "virology"],
            "pharmacy": ["medicine", "drug", "prescription", "meds", "dispensary"],
            "physiotherapy": ["exercise", "rehabilitation", "therapy", "physical therapy"],
            "dietetics": ["nutrition", "diet plan", "diet", "food advice"],
            "transfusion_medicine": ["blood bank", "transfusion", "donation"],

            # Emergency / Critical Care / Surgical Units
            "emergency": ["casualty", "trauma", "urgent", "er", "accident"],
            "icu": ["intensive care", "critical care", "ventilator", "nicu", "picu"],
            "operation_theatre": ["surgery", "operation", "procedure", "ot"],
            "burn_and_plastic_surgery": ["burn", "plastic", "cosmetic", "bps", "plastic surgery"],

            # Special Units
            "neonatology": ["nicu", "newborn", "neonate", "baby care"],
            "obstetrics_and_gynaecology": ["obg", "obgy", "pregnancy", "labour", "delivery", "women’s health"]
        }
    
    def extract_medical_entities(self, query: str) -> dict:
        """Extract medical-specific entities"""
        medical_entities = defaultdict(list)
        
        # Extract using hospital patterns
        for entity_type, pattern in self.hospital_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                if value:
                    medical_entities[entity_type].append({
                        "value": value,
                        "source": "hospital_patterns",
                        "confidence": 1.0
                    })
        
        # Categorize symptoms and conditions
        words = query.lower().split()
        for i, word in enumerate(words):
            # Check symptom categories
            for category, symptoms in self.medical_knowledge.items():
                if category.endswith("_symptoms"):
                    for symptom in symptoms:
                        if symptom in " ".join(words[i:i+3]):
                            medical_entities["symptoms"].append({
                                "value": symptom,
                                "category": category,
                                "confidence": 0.8
                            })
            
            # Check disease categories
            for category in ["chronic_conditions", "acute_conditions", "emergency_conditions"]:
                for condition in self.medical_knowledge[category]:
                    if condition in " ".join(words[i:i+3]):
                        medical_entities["conditions"].append({
                            "value": condition,
                            "category": category,
                            "confidence": 0.8
                        })
        
        return dict(medical_entities)
    
    def map_to_department(self, entities: dict, query: str) -> list:
        """Map entities to relevant departments"""
        relevant_departments = defaultdict(float)
        
        # Check symptoms and conditions
        for entity_type in ["symptoms", "conditions"]:
            if entity_type in entities:
                for entity in entities[entity_type]:
                    value = entity["value"].lower()
                    # Check each department's keywords
                    for dept, keywords in self.department_mapping.items():
                        if any(keyword in value for keyword in keywords):
                            relevant_departments[dept] += entity.get("confidence", 0.8)
        
        # Check medical specialties
        for specialty, keywords in self.medical_knowledge["specialties"].items():
            if any(keyword in query.lower() for keyword in keywords):
                relevant_departments[specialty] += 0.9
        
        # Sort by relevance score
        sorted_departments = sorted(
            relevant_departments.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"department": dept, "confidence": score}
            for dept, score in sorted_departments
            if score > 0.5
        ]
    
    def classify_medical_intent(self, query: str) -> tuple:
        """Classify medical-specific intent"""
        # First try parent's intent classification
        base_intent, base_confidence = super().classify_intent(query)
        
        # If confidence is low or intent is general, try hospital-specific intents
        if base_confidence < 0.6 or base_intent in ["general_information", "out_of_scope"]:
            result = self.intent_classifier(
                query,
                candidate_labels=self.hospital_specific_intents,
                hypothesis_template="This is a query about {}.",
                multi_label=False
            )
            
            hospital_intent = result["labels"][0]
            hospital_confidence = result["scores"][0]
            
            if hospital_confidence > base_confidence:
                return hospital_intent, hospital_confidence
        
        return base_intent, base_confidence
    
    def extract_entities(self, query: str) -> dict:
        """Enhanced entity extraction with medical focus"""
        # Get base entities
        base_entities = super().extract_entities(query)
        
        # Get medical-specific entities
        medical_entities = self.extract_medical_entities(query)
        
        # Merge entities
        merged_entities = base_entities.copy()
        for entity_type, entities in medical_entities.items():
            if entity_type not in merged_entities:
                merged_entities[entity_type] = entities
            else:
                merged_entities[entity_type].extend(entities)
        
        # Map to departments if needed
        if any(key in merged_entities for key in ["symptoms", "conditions"]):
            relevant_departments = self.map_to_department(merged_entities)
            if relevant_departments:
                merged_entities["suggested_departments"] = relevant_departments
        
        return merged_entities


def build_spacy_phrase_matcher():
    global phrase_matcher
    if not nlp_spacy or not phrase_matcher:
        logger.warning("spaCy or PhraseMatcher not initialized.")
        return

    phrase_matcher.clear()  # Avoid duplicate patterns

    for category, phrases in data_loader.all_known_entities.items():
        try:
            valid_phrases = [
                phrase for phrase in phrases 
                if isinstance(phrase, str) and len(phrase.strip()) > 2
            ]
            patterns = [nlp_spacy.make_doc(phrase.strip()) for phrase in valid_phrases]
            phrase_matcher.add(category, patterns)
        except Exception as e:
            logger.warning(f"Failed to add patterns for category '{category}': {e}")
    
    logger.info(f"spaCy PhraseMatcher loaded with categories: {list(data_loader.all_known_entities.keys())}")


shared_minilm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_models = {
    "multilingual": HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "qa": HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"),
    "general": shared_minilm,
    "ranking": HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5"),
    "hybrid": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"), 
    "factual": HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2"),
}
embedding = embedding_models["multilingual"]

def spacy_sentence_splitter(text: str) -> list:
    """Split long hospital paragraphs into spaCy-detected sentences."""
    doc = nlp_spacy(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


faiss_index_cache = {}
def initialize_faiss(embedding_model_instance=None):
    global faiss_index_cache
    effective_embedding_model = embedding_model_instance if embedding_model_instance else embedding
    model_id = effective_embedding_model.model_name # Ensure model_name attribute exists

    if model_id in faiss_index_cache:
        logger.info(f"Using cached FAISS index for model: {model_id}")
        return faiss_index_cache[model_id]

    faiss_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")


    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        try:
            logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH} using model: {model_id}")
            db_faiss = FAISS.load_local(FAISS_INDEX_PATH, effective_embedding_model, allow_dangerous_deserialization=True)
            faiss_index_cache[model_id] = db_faiss
            return db_faiss
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}. Rebuilding.")
            if os.path.exists(FAISS_INDEX_PATH):
                shutil.rmtree(FAISS_INDEX_PATH)
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    logger.info("Building new FAISS index for hospital data.")
    docs_for_faiss = prepare_documents()
    if not docs_for_faiss:
        logger.error("No documents prepared. FAISS index cannot be built.")
        return None

    chunked_docs = []
    for doc in docs_for_faiss:
        sentences = spacy_sentence_splitter(doc.page_content)
        
        # Join sentences into chunks of ~2-4 sentences (up to 1000 chars)
        current_chunk = []
        current_length = 0
        for sent in sentences:
            current_chunk.append(sent)
            current_length += len(sent)
            if current_length > 800:
                merged = " ".join(current_chunk)
                chunked_docs.append(Document(page_content=merged, metadata=doc.metadata))
                current_chunk, current_length = [], 0

        # Flush remaining
        if current_chunk:
            merged = " ".join(current_chunk)
            chunked_docs.append(Document(page_content=merged, metadata=doc.metadata))

    try:
        db_faiss = FAISS.from_documents(chunked_docs, effective_embedding_model)
        db_faiss.save_local(FAISS_INDEX_PATH)
        faiss_index_cache[model_id] = db_faiss
        logger.info(f"FAISS index built and saved to {FAISS_INDEX_PATH} with {len(chunked_docs)} chunks.")
        return db_faiss
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None

db_faiss = initialize_faiss()

try:
    fast_reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512) # Lightweight,Fast, multilingual, good for first pass
    strong_reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-base-v1', max_length=512) # Heavier, high accuracy, mxbai-rerank-large-v1
    # multilingual_reranker = CrossEncoder('Alibaba-NLP/gte-multilingual-reranker-base', max_length=512, device=device)
    logger.info("Loaded primary rerankers (BAAI & MXBAI) successfully.")
except Exception as e:
    logger.warning(f"Primary reranker load failed: {e}. Falling back to MiniLM models.")
    try:
        fast_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        strong_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
        logger.info("Loaded MiniLM rerankers as fallback.")
    except Exception as e2:
        logger.error(f"Fallback reranker load failed: {e2}. Rerankers unavailable.")
        fast_reranker, strong_reranker = None, None

def initialize_bm25():
    global bm25_corpus_docs, bm25_tokenized_corpus
    documents = prepare_documents() # Uses hospital data now
    if not documents:
        logger.error("No documents available for BM25 initialization.")
        return None

    bm25_corpus_docs = []
    bm25_tokenized_corpus = []

    for doc in documents:
        # Use spaCy sentence splitter here
        sentences = spacy_sentence_splitter(doc.page_content)
        for sent in sentences:
            if len(sent) > 30:  # skip short trivial lines
                bm25_corpus_docs.append(Document(page_content=sent, metadata=doc.metadata))
                bm25_tokenized_corpus.append(sent.lower().split())

    if not bm25_tokenized_corpus:
        logger.error("BM25 tokenized corpus is empty.")
        return None

    logger.info(f"Initialized BM25 with {len(bm25_tokenized_corpus)} documents for hospital data.")
    return BM25Okapi(bm25_tokenized_corpus)
bm25 = initialize_bm25()

# Initialize NLU Processor
nlu_processor = EnhancedNLUProcessor()

def get_embedding_model_for_query(query):
    query_lower = query.lower()

    # Hybrid model: Broad/general search or fuzzy context
    if any(word in query_lower for word in ["search", "find", "nearby", "available", "locate", "which room", "what rooms", "where can i find", "provide"]):
        return embedding_models["hybrid"]
    
    # QA-optimized: Direct factual questions
    if any(word in query_lower for word in ["what is", "who is", "where is", "define", "tell me about", "doctor", "department"]):
        return embedding_models["qa"]
    
    # Factual: Explanation or in-depth info
    if any(word in query_lower for word in ["explain", "describe", "details about", "how does", "procedure", "treatment", "everything about"]):
        return embedding_models["factual"]
    
    # Ranking: Comparisons or list-based queries
    if any(word in query_lower for word in ["list all", "compare services", "rank doctors", "best", "vs", "versus"]):
        return embedding_models["ranking"]
    
    # Multilingual: Default fallback, good multilingual support
    else:
        model = embedding_models["multilingual"]
    logger.info(f"[Embedding Model Routing] Using: {model.model_name} for query: {query}")
    return model


def extract_formatting_instructions(query: str) -> dict:
    """Extract formatting instructions from the query."""
    instructions = {}
    
    # Format patterns
    format_patterns = {
        "bullet_points": [
            r"\b(?:in|as|with|using)\s+bullet\s*points?\b",
            r"\bbullet\s*(?:list|format)\b",
            r"\blist\s+format\b"
        ],
        "table_format": [
            r"\b(?:in|as|with|using)\s+(?:a\s+)?table\b",
            r"\btabular\s*format\b"
        ],
        "step_by_step": [
            r"\b(?:step\s+by\s+step)\b",
            r"\b(?:in|as|with)\s+steps\b"
        ]
    }
    
    query_lower = query.lower()
    
    for format_type, patterns in format_patterns.items():
        if any(re.search(pattern, query_lower) for pattern in patterns):
            instructions["format"] = format_type
            break
            
    return instructions

def clean_extracted_entities(entities):
    """
    Cleans broken or subword tokens from entity lists like doctors/persons/etc.
    - Removes '##' prefixes (BERT-style subwords)
    - Joins fragments into full names (e.g., 'Shr', '##uti', 'Sharma' → 'Shruti Sharma')
    - Handles prefixes like 'Dr.' or 'Prof.'
    - Deduplicates and trims
    """
    name_prefixes = {"dr", "dr.", "prof", "prof.", "mr", "ms", "mrs"}
    # Formatting related terms that should not be treated as entities
    formatting_terms = {
        "bullet", "points", "bullet points", "table", "format", "list", 
        "step by step", "steps", "tabular"
    }

    cleaned_entities = {}
    for key, values in entities.items():
        if not values or not isinstance(values, list):
            cleaned_entities[key] = []
            continue

        tokens = [v.replace("##", "").strip() for v in values if isinstance(v, str) and len(v.strip()) > 0]
        phrases = []
        buffer = []

        for token in tokens:
            token_lower = token.lower()
            
            # Skip formatting-related terms
            if token_lower in formatting_terms:
                continue

            if token_lower in name_prefixes:
                if buffer:
                    phrases.append(" ".join(buffer))
                buffer = [token]
            elif token[0].isupper() or token.istitle():
                buffer.append(token)
            else:
                if buffer:
                    buffer.append(token)
                else:
                    buffer = [token]

        if buffer:
            phrases.append(" ".join(buffer))

        cleaned = sorted(set([p.strip() for p in phrases if p.strip()]))
        cleaned_entities[key] = cleaned

    return cleaned_entities

def extract_spacy_ner_entities(text: str):
    if not nlp_spacy:
        return {}

    doc = nlp_spacy(text)
    entity_map = {"doctors": [], "departments": [], "locations": [], "dates": [], "times": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entity_map["doctors"].append(ent.text)
        elif ent.label_ in {"ORG"}:
            entity_map["departments"].append(ent.text)
        elif ent.label_ in {"GPE", "LOC"}:
            entity_map["locations"].append(ent.text)
        elif ent.label_ == "DATE":
            entity_map["dates"].append(ent.text)
        elif ent.label_ == "TIME":
            entity_map["times"].append(ent.text)

    # Clean + dedupe
    for k in entity_map:
        entity_map[k] = sorted(set(e.strip() for e in entity_map[k] if len(e.strip()) > 1))
    return entity_map

def merge_entities(base: dict, extra: dict):
    merged = base.copy()
    for key, val in extra.items():
        if key not in merged:
            merged[key] = val
        else:
            merged[key].extend(val)
            merged[key] = sorted(set(merged[key]))  # Dedup
    return merged


import time
def refresh_faiss_and_bm25():
    global db_faiss, bm25, faiss_index_cache, data_loader
    logger.info("Refreshing FAISS index and BM25 model for hospital data.")

    # Reload data first
    #data_loader = HospitalDataLoader(hospital_filepath=HOSPITAL_MODEL_JSON_PATH, qa_filepath=QA_PAIRS_JSON_PATH) # Ensure it reloads hospital data

    faiss_index_cache.clear()
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"Failed to delete FAISS index path: {e}")

    db_faiss = initialize_faiss() 
    bm25 = initialize_bm25() 

    if db_faiss and bm25:
        logger.info("FAISS and BM25 refreshed successfully for hospital data.")
    else:
        logger.error("Hospital data refresh failed for FAISS or BM25.")
    return db_faiss, bm25

def detect_query_complexity(query):
    query_lower = query.lower()
    token_count = len(query.split())

    if any(conj in query_lower for conj in [" and ", " or ", " but also ", " as well as "]) and token_count > 7: 
        return "complex"
    if any(phrase in query_lower for phrase in [
        "list all services", "all doctors in department", "compare procedures", 
        "explain treatment options", "compare departments", "show all rooms"
    ]):
        return "complex"
    if token_count <= 5 and any(q_word in query_lower for q_word in ["where is room", "dr. email", "phone for cardiology"]):
        return "simple"
    return "normal"

def hybrid_retriever(query, k_simple=5, k_normal=8, k_complex=12, override_k=None):
    selected_embedding_instance = get_embedding_model_for_query(query)

    current_db_faiss = (
        faiss_index_cache.get(selected_embedding_instance.model_name) 
        or initialize_faiss(selected_embedding_instance)
    )

    if not current_db_faiss:
        logger.error("FAISS database (hospital) not available.")
        return []

    if not bm25:
        logger.error("BM25 (hospital) not available. Falling back to FAISS only.")
        return current_db_faiss.as_retriever(search_kwargs={"k": k_normal}).get_relevant_documents(query)

    complexity = detect_query_complexity(query)
    k_val = override_k if override_k is not None else (
        k_simple if complexity == "simple" else (k_normal if complexity == "normal" else k_complex)
    )

    logger.info(f"Hybrid retrieval (hospital) for '{query}' → complexity: {complexity}, k={k_val}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(
            current_db_faiss.as_retriever(search_kwargs={"k": k_val}).get_relevant_documents, query
        )
        bm25_future = executor.submit(bm25_retriever_func, query, k_val)

        try:
            faiss_docs = faiss_future.result(timeout=10)
            bm25_top_docs = bm25_future.result(timeout=10)
        except TimeoutError:
            logger.warning("Retrieval timed out (hospital).")
            faiss_docs = faiss_future.result() if faiss_future.done() else []
            bm25_top_docs = bm25_future.result() if bm25_future.done() else []
        except Exception as e:
            logger.error(f"Error in hospital data retrieval: {e}")
            faiss_docs, bm25_top_docs = [], []

    all_docs_dict = {doc.page_content: doc for doc in faiss_docs}
    for doc in bm25_top_docs:
        if doc.page_content not in all_docs_dict:
            all_docs_dict[doc.page_content] = doc

    merged_docs = list(all_docs_dict.values())
    logger.info(f"Hybrid merged {len(merged_docs)} docs (hospital).")
    return merged_docs[:k_val * 2] # Return up to 2*k_val


# Utility to create a hashable cache key from query + doc IDs
def _create_rerank_cache_key(query: str, docs: list) -> str:
    doc_ids = [doc.metadata.get("source_doc_id", doc.page_content[:30]) for doc in docs]
    key_str = query + "::" + "::".join(doc_ids)
    return hashlib.md5(key_str.encode()).hexdigest()

# Actual cache
_rerank_fast_cache = {}
# Fast Reranker (first stage)
def rerank_documents_fast(query, docs, top_k=8):
    if not fast_reranker:
        logger.warning("Fast reranker not available. Returning top documents without reranking.")
        return docs[:top_k]

    if not docs or len(docs) < 2:
        logger.info("Not enough documents for reranking. Returning as-is.")
        return docs[:top_k]

    try:
        cache_key = _create_rerank_cache_key(query, docs)
        if cache_key in _rerank_fast_cache:
            logger.info(f"Using cached fast reranked docs for query: '{query}'")
            return _rerank_fast_cache[cache_key][:top_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = fast_reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:top_k]]

        _rerank_fast_cache[cache_key] = top_docs
        logger.info(f"Cached fast reranked docs for query: '{query}' (top_k={top_k})")
        return top_docs

    except Exception as e:
        logger.error(f"Error during fast reranking: {e}. Returning top docs.")
        return docs[:top_k]

# Strong Reranker (final stage)
def rerank_documents_strong(query, docs, top_k=3):
    if not strong_reranker:
        logger.warning("strong reranker not available. Returning top documents without reranking.")
        return docs[:top_k]

    if not docs or len(docs) < 2:
        logger.info("Not enough documents for reranking. Returning as-is.")
        return docs[:top_k]

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = strong_reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.info(f"strong reranker reranked {len(docs)} docs, returning top {top_k}.")
        return [doc for score, doc in scored_docs[:top_k]]
    except Exception as e:
        logger.error(f"Error during strong reranker reranking: {e}. Returning top docs.")
        return docs[:top_k]
    
# 1. Build a corpus of canonical medical terms you care about
CANONICAL_TERMS = [
    "cardiology", "neurology", "orthopedics", "intensive care unit", "emergency room",
    "appointment booking", "operating hours", "contact information", "Anesthesia" # …
]

emb_model = SentenceTransformer("pritamdeka/S-Biomed-Roberta-snli-multinli-stsb")

term_embeddings = emb_model.encode(CANONICAL_TERMS, normalize_embeddings=True)

SYNONYM_MAP = {
    # Facilities & Navigation
    "elevator": ["lift"], "lift": ["elevator"],
    "toilet": ["washroom", "restroom", "lavatory", "wc"],
    "stairs": ["staircase"], "staircase": ["stairs"],
    "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0"], "gf": ["floor 0"],
    "building": ["hospital building"], "hospital building": ["building"],
    "waiting room": ["waiting area"], "waiting area": ["waiting room"],
    "ward": ["room"], "room": ["ward"],

    # Contact & Timings
    "contactno": ["contact number", "phone no", "phone number", "contact"],
    "timings": ["operating hours", "open hours", "availability", "schedule"],
    "schedule": ["timings"],

    # People
    "doctor": ["dr", "dr.", "physician", "consultant", "specialist", "professor", "doc"],
    "dr.": ["doctor"], "physician": ["doctor"], "doc": ["doctor"],
    "staff": ["hospital staff"],

    "thopedics": ["bone department", "ortho"], "bones": ["orthopedics"],
    "pediatrics": ["child care", "paediatrics", "peds"], "children": ["pediatrics"],
    "pediatric surgery": ["child surgery", "paediatric surgery"],

    "radiology": ["x-ray department", "imaging"], "scan": ["radiology"],
    "radiotherapy": ["radiation therapy", "cancer radiation", "radio therapy"],

    "dermatology": ["skin"], "skin": ["dermatology"],
    "gynecology": ["women", "obgyn"], "women": ["gynecology"],
    "ophthalmology": ["eye", "eye department", "eye care"], "eye": ["ophthalmology"],

    "psychiatry": ["psychiatrist"], "psychiatrist": ["psychiatry"],
    "oncology": ["cancer"], "cancer": ["oncology"],
    "anesthesia": ["anesthesiology"], "anesthesiology": ["anesthesia"],
    "pathology": ["lab test", "specimen lab", "path lab"],

    "orthodontics": ["dental braces", "teeth alignment", "braces doctor"],
    "nephrology": ["kidney specialist", "renal doctor"], "kidney": ["nephrology"],

    "neurosurgery": ["brain surgery", "neuro surgeon", "spine surgery"],

    # Services & Booking
    "checkup": ["diagnosis"], "test": ["diagnosis"],
    "lab": ["laboratory"], "laboratory": ["lab"],
    "mri scan": ["mri"], "ct scan": ["ct"], "ultrasound scan": ["ultrasound"],
    "report": ["diagnosis report"],
    "appointment": ["booking", "reservation", "appt"], "booking": ["appointment"],

    # Equipment
    "ventilator machine": ["ventilator"], "ventilator": ["ventilator machine"],
    "wheel chair": ["wheelchair"], "wheelchair": ["wheel chair"],

    # Institution
    "aiims": ["all india institute of medical sciences"],
    "all india institute of medical sciences": ["aiims"]
}

lru_cache(maxsize=1024)
def _semantic_expand(query: str, top_k: int = 3, sim_threshold: float = 0.55) -> list:
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    sims = util.cos_sim(q_emb, term_embeddings)[0].cpu().numpy()
    top_idx = np.argpartition(-sims, range(top_k))[:top_k]
    return [
        CANONICAL_TERMS[i]
        for i in top_idx
        if sims[i] >= sim_threshold and CANONICAL_TERMS[i] not in query.lower()
    ]

def expand_query_with_synonyms(query: str, max_variants: int = 6):
    query_lower = query.lower()
    variants = {query_lower}

    logger.debug(f"[SynonymExpansion] Starting with query: '{query}'")

    # 1. Dictionary based synonyms expansion 
    for keyword, synonyms in SYNONYM_MAP.items():
        all_forms = [keyword] + synonyms
        for current_variant in list(variants):  # iterate on a snapshot
            for form in all_forms:
                if re.search(rf'\b{re.escape(form)}\b', current_variant):
                    for repl in all_forms:
                        if repl.lower() != form.lower():
                            new_variant = re.sub(rf'\b{re.escape(form)}\b', repl, current_variant)
                            variants.add(new_variant)

    logger.debug(f"[SynonymExpansion] After dictionary expansion: {variants}")

    # 2. Semantic expansion if few variants
    if len(variants) <= 2:
        sem_exp = _semantic_expand(query_lower)
        variants.update(sem_exp)
        logger.debug(f"[SynonymExpansion] After semantic expansion: {variants}")

    # 3. Trim / log
    variants = {v.strip() for v in variants if len(v.strip()) > 3}
    final = list(variants)[:max_variants]

    if len(final) > 1 and query_lower not in final:
        logger.info(f"[SynonymExpansion-Final] '{query}' → Variants: {final}")
    else:
        logger.debug(f"[SynonymExpansion-Final] No significant expansion for: '{query}' → {final}")
    return final


def bm25_retriever_func(query, k=10): # Logic remains the same, uses hospital corpus
    if not bm25 or not bm25_corpus_docs: logger.warning("BM25 model or hospital corpus not initialized."); return []
    expanded_queries = expand_query_with_synonyms(query); all_scored_docs = {}
    for q_variant in expanded_queries:
        tokenized_query = q_variant.lower().split()
        if not tokenized_query: continue
        try:
            scores = bm25.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                if score > 0:
                    all_scored_docs[i] = max(all_scored_docs.get(i, 0.0), score)
        except Exception as e: 
            logger.error(f"Error getting BM25 scores for variant '{q_variant}': {e}"); continue
    
    valid_indices = [item_index for item_index in all_scored_docs.keys() if item_index < len(bm25_corpus_docs)]
    sorted_indices = sorted(valid_indices, key=lambda i: all_scored_docs[i], reverse=True)
    top_docs = [bm25_corpus_docs[i] for i in sorted_indices[:k]]
    logger.info(f"BM25 (hospital) retrieved {len(top_docs)} docs for query '{query}'.")
    return top_docs

def detect_target_language_for_response(query):
    language_map = {"hindi": "hi", "punjabi": "pa", "tamil": "ta", "telugu": "te", "kannada": "kn", "marathi": "mr", "bengali": "bn", "urdu": "ur", "gujarati": "gu", "malayalam": "ml", "english": "en", "spanish": "es", "french": "fr", "german": "de", "russian": "ru"}
    query_lower = query.lower()
    for lang_name, lang_code in language_map.items():
        if re.search(rf'\bin\s+{re.escape(lang_name)}\b', query_lower):
            cleaned_query = re.sub(rf'\s*\bin\s+{re.escape(lang_name)}\b', '', query_lower, flags=re.IGNORECASE).strip()
            logger.info(f"Detected target response language: {lang_name} ({lang_code}). Cleaned query: '{cleaned_query}'")
            return cleaned_query, lang_code
    return query, None

def normalize_department_name(dept_name: str) -> str:
    """Normalize department names to standard forms."""
    dept_name = dept_name.lower().strip()
    
    # Department name standardization mappings
    standard_names = {
        "anaesthesia": "anesthesia",
        "anaesthesiology": "anesthesia",
        "anesthesiology": "anesthesia",
        "cardio": "cardiology",
        "cardiology": "cardiology",
        "neuro": "neurology",
        "neurology": "neurology",
        "urology": "urology",
        "ortho": "orthopedics",
        "orthopaedics": "orthopedics",
        "orthopedics": "orthopedics",
        "paediatrics": "pediatrics",
        "pediatrics": "pediatrics",
        "obgyn": "gynecology",
        "gynaecology": "gynecology",
        "gynecology": "gynecology",
        "opthalmology": "ophthalmology",
        "ophthalmology": "ophthalmology",
        "ent": "ent",
        "otolaryngology": "ent",
        "psych": "psychiatry",
        "psychiatry": "psychiatry"
    }
    
    # Check for exact matches first
    if dept_name in standard_names:
        return standard_names[dept_name]
    
    # Check for partial matches
    for variant, standard in standard_names.items():
        if variant in dept_name:
            return standard
            
    return dept_name

def extract_entities_rule_based(query):
    """Extract entities using rule-based patterns with enhanced department handling."""
    entities = {}
    query_lower = query.lower()

    # Department name mappings and variations
    department_mappings = {
        "anesthesia": ["anesthesia", "anaesthesia", "anesthesiology", "anaesthesiology"],
        "cardiology": ["cardiology", "cardiac", "heart", "cardio"],
        "neurology": ["neurology", "neuro", "brain", "nerve"],
        "orthopedics": ["orthopedics", "orthopaedics", "ortho", "bone"],
        "pediatrics": ["pediatrics", "paediatrics", "child", "children"],
        "radiology": ["radiology", "imaging", "x-ray", "scan"],
        "surgery": ["surgery", "surgical", "operation"],
        "emergency": ["emergency", "er", "accident", "trauma"],
        "ophthalmology": ["ophthalmology", "eye", "ophthal"],
        "ent": ["ent", "ear nose throat", "otolaryngology"],
        "dermatology": ["dermatology", "skin"],
        "psychiatry": ["psychiatry", "mental health", "psych"],
        "dental": ["dental", "dentistry", "teeth"],
        "gynecology": ["gynecology", "gynaecology", "obgyn", "obstetrics"],
        "oncology": ["oncology", "cancer"],
        "urology": ["urology", "urinary"],
        "pathology": ["pathology", "lab", "laboratory"],
        "physiotherapy": ["physiotherapy", "physical therapy", "rehab"],
        "nutrition": ["nutrition", "diet"],
        "pharmacy": ["pharmacy", "medicine", "drug"]
    }

    # Common department indicators and patterns
    dept_indicators = [
        "department", "dept", "ward", "unit", "center", "centre", "clinic", "wing",
        "section", "division", "specialty", "speciality"
    ]
    
    # Extract departments with enhanced pattern matching
    found_departments = []
    
    # First, check for department name with indicators
    for dept, variations in department_mappings.items():
        for variation in variations:
            # Check exact matches
            if variation in query_lower:
                found_departments.append(dept)
                break
                
            # Check with indicators
            for indicator in dept_indicators:
                patterns = [
                    f"{variation} {indicator}",
                    f"{indicator} of {variation}",
                    f"{variation}-{indicator}",
                    f"{indicator} {variation}"
                ]
                if any(pattern in query_lower for pattern in patterns):
                    found_departments.append(dept)
                    break
                    
            if dept in found_departments:
                break
    
    # Handle special cases and common phrases
    dept_phrases = [
        (r"(?:show|tell|list|give|get|find).*(?:doctors?|physicians?|specialists?).*(?:from|in|at|of)\s+(\w+)", 1),
        (r"(\w+)\s+(?:doctors?|physicians?|specialists?)", 1),
        (r"(\w+)\s+(?:department|dept|ward|unit)", 1)
    ]
    
    for pattern, group in dept_phrases:
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            dept_name = match.group(group)
            # Normalize and validate department name
            normalized_dept = normalize_department_name(dept_name)
            for dept, variations in department_mappings.items():
                if normalized_dept in variations or normalized_dept == dept:
                    found_departments.append(dept)
                    break

    if found_departments:
        entities["departments"] = list(set(found_departments))

    # Extract doctor-related information
    doctor_patterns = [
        r"(?:dr\.?|doctor|prof\.?|professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Proper name format
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:md|mbbs|do|phd)",  # Name followed by qualification
    ]
    
    found_doctors = []
    common_words = {"in", "at", "on", "the", "and", "or", "for", "with", "bullet", "points"}
    
    for pattern in doctor_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)  # Note: removed lower() to preserve capitalization
        for match in matches:
            doctor_name = match.group(1).strip()
            # Validate the name:
            # 1. Check if it's not just common words
            words = set(doctor_name.lower().split())
            if not words.issubset(common_words):
                # 2. Ensure it looks like a proper name (first letter capitalized)
                name_parts = doctor_name.split()
                if all(part[0].isupper() for part in name_parts):
                    found_doctors.append(doctor_name)
    
    if found_doctors:
        entities["doctors"] = list(set(found_doctors))

    # Add number of doctors requested if specified
    number_pattern = r"(?:any|show|tell|list|give)\s+(\d+)\s+doctors?"
    number_match = re.search(number_pattern, query_lower)
    if number_match:
        entities["requested_count"] = int(number_match.group(1))

    return entities

from collections import defaultdict

def extract_entities_spacy(query: str) -> dict:
    if not nlp_spacy or not phrase_matcher:
        logger.warning("spaCy model or PhraseMatcher not loaded. Skipping spaCy entity extraction.")
        return {}

    doc = nlp_spacy(query)
    matches = phrase_matcher(doc)
    entities = defaultdict(list)

    for match_id, start, end in matches:
        label = nlp_spacy.vocab.strings[match_id]
        span = doc[start:end].text.strip()
        if span:
            entities[label].append(span)

    # Deduplicate and sort
    for k in entities:
        entities[k] = sorted(set(entities[k]))

    logger.info(f"[spaCy NER] Extracted entities: {dict(entities)}")
    return dict(entities)

def resolve_pronouns_with_spacy(text: str, fallback_entity: str = "") -> str:
            if not nlp_spacy or not fallback_entity:
                return text  # Safe fallback

            doc = nlp_spacy(text)
            rewritten = text
            replacements = {}

            for token in doc:
                if token.lower_ in {"he", "she", "his", "her", "they", "their", "it", "its", "this", "that"}:
                    if token.dep_ in {"nsubj", "poss", "dobj", "attr", "pobj"}:
                        rep = fallback_entity
                        if token.tag_ in {"PRP$"} or token.lower_ in {"his", "her", "their", "its"}:
                            rep += "'s"
                        replacements[token.text] = rep

            for pronoun, replacement in replacements.items():
                rewritten = re.sub(rf"\b{re.escape(pronoun)}\b", replacement, rewritten, flags=re.IGNORECASE)

            if rewritten != text:
                logger.info(f"[Coref Rewrite - spaCy] '{text}' → '{rewritten}' using spaCy dependencies")
            return rewritten

semantic_matcher_model = shared_minilm

def semantic_similarity(a: str, b: str) -> float:
    embeddings = semantic_matcher_model.encode([a, b], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


def ground_entity_to_docs(entity_value: str, docs, threshold=0.7):
    """
    Ground a single entity against metadata & content of retrieved docs.
    Returns:
        best_score: float
        best_match_doc: Document or None
        matched_field: str or None
    Logs match scores for debugging.
    """
    best_score = 0
    best_doc = None
    matched_field = None
    all_scores_log = []  # For debug traceability

    for doc_index, doc in enumerate(docs):
        fields_to_check = list(doc.metadata.keys()) + ["page_content"]
        for field in fields_to_check:
            val = doc.metadata.get(field) if field != "page_content" else doc.page_content
            if not isinstance(val, str):
                continue
            try:
                score = semantic_similarity(entity_value, val)
                all_scores_log.append({
                    "entity": entity_value,
                    "score": round(score, 3),
                    "field": field,
                    "doc_id": doc.metadata.get("source_doc_id", f"doc_{doc_index}")
                })
                if score > best_score:
                    best_score = score
                    best_doc = doc
                    matched_field = field
            except Exception as e:
                logger.warning(f"[Grounding Error] Field: '{field}' → {e}")

    # Log top matches (optional: only log if best_score is low)
    if best_score < threshold:
        logger.warning(f"[Entity Grounding Failed] '{entity_value}' best_score={round(best_score, 3)} < threshold={threshold}")
        logger.info(f"[Grounding Candidates for '{entity_value}']:\n" +
            "\n".join(
                f"DocID={item['doc_id']} | Field={item['field']} | Score={item['score']}" 
                for item in sorted(all_scores_log, key=lambda x: -x["score"])[:5]
            )
        )

    return best_score, best_doc, matched_field#, all_scores_log

def store_summary_to_faiss(user_id, memory):
    try:
        summary = memory.buffer.strip()
        if not summary:
            logger.info("No summary to store for user.")
            return

        doc = Document(
            page_content=summary,
            metadata={
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        chat_memory_faiss.add_documents([doc])
        chat_memory_faiss.save_local(CHAT_MEMORY_INDEX_PATH)
        logger.info(f"Stored chat memory summary for user {user_id} in FAISS.")
    except Exception as e:
        logger.error(f"Error storing chat memory: {e}")


llm_task_classifier = ChatGroq(
    temperature=0.0,
    model="llama3-8b-8192",
)
ALLOWED_TASK_LABELS = {
    "listing_all", "listing_specific", "location", "contact_info", "booking_info",
    "explanation", "comparison", "operating_hours", "doctor_availability",
    "department_info", "service_info", "out_of_scope", "general_information"
}

def detect_task_type_llm(query: str) -> str:
    system_prompt = (
        "You are a hospital domain classifier. Given a user query, classify it into one of these task types:\n\n"
        + "\n".join(f"- {label}" for label in ALLOWED_TASK_LABELS) +
        "\n\nOnly return one of the above labels as a response. Don't explain."
    )
    response = llm_task_classifier.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    label = response.content.strip()
    return label if label in ALLOWED_TASK_LABELS else "general_information"


def detect_task_type_rule_based(query): # Adapted for hospital
    query_l = query.lower()
    if any(kw in query_l for kw in ["list all", "show all doctors", "all departments", "every service"]): 
        return "listing_all"
    if any(kw in query_l for kw in ["list services", "summarize treatments", "overview of doctors"]): 
        return "listing_specific"
    if any(kw in query_l for kw in ["where is", "location of", "find near", "how to reach", "direction to", "room number", "which floor"]): 
        return "location"
    if any(kw in query_l for kw in ["email of", "contact for", "phone number of", "call dr", "website for hospital"]): 
        return "contact_info"
    if any(kw in query_l for kw in ["book appointment", "appointment with", "availability of doctor", "reserve slot", "schedule visit"]): 
        return "booking_info" # For appointments
    if any(kw in query_l for kw in ["how to ", "explain procedure", "what are symptoms", "details about disease", "treatment for"]): 
        return "explanation"
    if any(kw in query_l for kw in ["compare treatments", "difference between doctors", "service A vs service B"]): 
        return "comparison"
    if any(kw in query_l for kw in ["operating hours", "timings", "when is opd open", "doctor schedule", "visiting hours"]): 
        return "operating_hours" # or doctor_availability
    if any(kw in query_l for kw in ["doctor availability", "is dr available", "dr schedule"]): 
        return "doctor_availability"
    if any(kw in query_l for kw in ["department of", "cardiology services", "info on neurology dept"]): 
        return "department_info"
    if any(kw in query_l for kw in ["service offered", "x-ray available", "mri cost info"]): 
        return "service_info"
    if any(kw in query_l for kw in [
        "book appointment", "make appointment", "dr appointment", "appointment slot", "book slot", "i want to meet", "need to see doctor"
    ]): 
        return "booking_info"
    # Out of scope - keep general
    if any(kw in query_l for kw in ["weather", "time now", "news", "stock price", "meaning of life", "who are you", "are you real", "do you sleep", "who created you", "what is your name"]): 
        return "out_of_scope"
    
    return "general_information" # Default


@lru_cache(maxsize=500)
def detect_task_type_llm_cached(query: str) -> str:
    return detect_task_type_llm(query)

def detect_task_type(query: str) -> str:
    task_by_rule = detect_task_type_rule_based(query)
    if task_by_rule != "general_information":
        return task_by_rule
    return detect_task_type_llm(query)


def extract_doctor_name(text: str) -> str:
    # 1. Try regex (fast, low-resource)
    match = re.search(r"(dr\.?\s+[a-zA-Z]+\s+[a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Fallback to NER
    doc_ents = nlu_processor.extract_entities(text)
    if "PER" in doc_ents:
        for name in doc_ents["PER"]:
            if len(name.split()) >= 2:
                return name

    return ""

def format_doctor_response(doc: dict) -> str:
    try:
        availability = format_doctor_availability(doc.get("availability", {}))
        return f"""
Doctor: {doc.get('name', 'N/A')}
Department: {doc.get('department', 'N/A')}
Designation: {doc.get('designation', 'N/A')}
Specialization: {doc.get('specialization', 'N/A')}
Availability: {availability}
Phone: {doc.get('phone', 'Not available')}
Email: {doc.get('email', 'Not available')}
Profile: {doc.get('profile', 'N/A')}
""".strip()
    except Exception as e:
        return f"[Error formatting doctor profile]: {e}"

def get_doctor_by_name(query_name: str, docs: list, threshold: int = 85) -> str:
    def norm(x: str) -> str:
        return x.lower().replace("dr ", "").replace("dr. ", "").strip()

    query_norm = norm(query_name)
    best_match = None
    best_score = 0

    for doc in docs:
        for doctor in doc.metadata.get("doctor_profiles", []):
            doc_name = doctor.get("name", "")
            doc_name_norm = norm(doc_name)
            score = fuzz.token_set_ratio(query_norm, doc_name_norm)
            if query_norm in doc_name_norm or score >= threshold:
                if score > best_score:
                    best_match = doctor
                    best_score = score

    if best_match:
        return format_doctor_response(best_match)
    return None


from symspellpy import SymSpell, Verbosity
# === Stopwords and entity prefixes to preserve ===
STOPWORDS = {
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "to", "of", "on", "in", "at", "by",
    "for", "with", "about", "as", "into", "like", "from", "than", "that", "this", "it"
}
ENTITY_STOPWORDS = {
    "dr", "dr.", "doctor", "prof", "aiims", "cardiology", "reema", "gaurav", "singh", "yadav"
}

# === Initialize SymSpell globally ===
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

try:
    dictionary_path = "resources/frequency_dictionary_en_82_765.txt"
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"{dictionary_path} not found.")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    logger.info(f"Loaded SymSpell dictionary from: {dictionary_path}")
except Exception as e:
    logger.warning(f"[SymSpell] Could not load base dictionary: {e}. Spell correction may be degraded.")

# === Inject hospital-specific terms ===
custom_terms = [
    "opd", "icu", "aiims", "xray", "ecg", "ot", "reema", "gaurav", "radiology",
    "cardiology", "orthopedics", "gynecology", "neurology", "dermatology", "dr", "dr."
]
for term in custom_terms:
    sym_spell.create_dictionary_entry(term.lower(), 10000)  # High frequency boosts match priority

def correct_spelling(text: str, verbose=False) -> str:
    try:
        detected_lang = detect(text)
        if detected_lang != "en":
            if verbose:
                print(f"[SpellCheck] Skipping correction for non-English text: {text} (lang: {detected_lang})")
            return text
    except Exception as e:
        logger.warning(f"[SpellCheck] Language detection failed: {e}. Proceeding with correction.")

    text_lower = text.lower().strip()

    # Try compound correction first
    suggestions = sym_spell.lookup_compound(text_lower, max_edit_distance=2)
    if suggestions and suggestions[0].term != text_lower:
        corrected_compound = suggestions[0].term
        if verbose:
            print(f"[SpellCheck] Compound corrected: '{text}' → '{corrected_compound}'")
        return corrected_compound

    # Word-level fallback
    corrected_tokens = []
    for token in text.split():
        token_clean = token.lower()
        if token_clean in STOPWORDS or token_clean in ENTITY_STOPWORDS:
            corrected_tokens.append(token)  # Preserve original casing
            continue

        word_suggestions = sym_spell.lookup(token_clean, Verbosity.CLOSEST, max_edit_distance=2)
        corrected = word_suggestions[0].term if word_suggestions else token
        corrected_tokens.append(corrected)

    corrected_text = " ".join(corrected_tokens)
    if verbose:
        print(f"[SpellCheck] Word-level corrected: '{text}' → '{corrected_text}'")
    return corrected_text


def collapse_repeated_letters(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1', text) # e.g. helllooo -> helo

def detect_conversational_intent(query):
    query_corrected = correct_spelling(query) # Correct spelling first
    query_clean = collapse_repeated_letters(query_corrected.lower().strip())

    greeting_variants = [
        "hi", "hello", "namaste", "hey", "greetings", "good morning", "good afternoon", "good evening", 
        "good night", "good day", "hiya", "yo", "hey there", "howdy", "salutations", "sup"
    ]
    exit_variants = [
        "bye", "goodbye", "see you", "take care", "farewell", "cya", "see ya", "later", 
        "talk to you later", "adios", "catch you later", "gotta go", "until next time",
        "i'm leaving", "that's all", "i'm done", "bye for now", "peace out", "okay bye", "exit"
    ]
    # Reduced smalltalk variants to be less aggressive, more for pure chit-chat
    smalltalk_variants = [
         "how are you", "how's it going", "what's up", "wassup", 
         "bored", "i'm back", "doing nothing", "tell me something", 
         "interesting", "just checking", "just saying hi",
         "hi again", "you awake?", "you online?", "mood off", "i'm tired", "i'm bored", 
         "anything new?", "say something", "tell me a joke", "reply pls", "pls respond",
         "ok", "okay", "cool", "sure", "fine", "great", "nice", "good", "awesome", "super" # Added common affirmations often part of small talk
    ]
    appreciation_variants = [
        "thank you", "thanks", "thx", "ty", "tysm", # Moved thanks here
        "you are doing good", "good job", "great work", "well done", "very well", 
        "appreciate it", "thanks a lot", "thank you so much", "that's helpful", 
        "amazing answer", "awesome reply", "you nailed it", "you're awesome", 
        "you rock", "brilliant", "excellent", "superb", "love that", "fantastic", 
        "mind blowing", "next level", "exactly what I needed", "so quick", "so smart"
    ]
    confirmation_variants = [
        "yes", "yep", "yeah", "sure", "absolutely", "of course", "definitely", 
        "yup", "you got it", "correct", "right", "exactly", "that's right", "alright", "indeed"
    ]
    negation_variants = [
        "no", "nope", "nah", "not really", "never", "i don't think so", "i dont think so",
        "wrong", "that's incorrect", "incorrect", "not correct", "not right"
    ]
    help_variants = [
        "i need your help", "can you help me", "help me", "i need assistance", "please help", "can you assist me",
        "could you help me", "would you help me", "i'm looking for help", "can you support me", "can you guide me",
        "need some help", "i require assistance", "please assist me", "help please", "i want your help",
        "need your support", "can you give me a hand", "i could use some help", "can you aid me", "i need a hand"
    ]

    def fuzzy_match(query_input, variant_list):
        for variant in variant_list:
            score = fuzz.partial_ratio(query_input, variant)
            logger.debug(f"[Fuzzy Match] '{query_input}' vs '{variant}' → score: {score}")
            if score >= 88 and abs(len(query_input) - len(variant)) <= 8:
                return True
        return False

    if fuzzy_match(query_clean, greeting_variants):
        return "greeting"
    if fuzzy_match(query_clean, exit_variants):
        return "exit"
    if fuzzy_match(query_clean, smalltalk_variants):
        return "smalltalk"
    if fuzzy_match(query_clean, appreciation_variants):
        return "appreciation"
    if fuzzy_match(query_clean, confirmation_variants):
        return "confirmation"
    if fuzzy_match(query_clean, negation_variants):
        return "negation"
    if fuzzy_match(query_clean, help_variants):
        return "help"
    # Fallback: LLM-based intent detection (only if fuzzy match fails)
    if USE_LLM_INTENT_FALLBACK:
        try:
            prompt = f"""
    You are an intent classifier for hospital chatbot. Categorize the user message strictly into one of the following:
    - greeting
    - exit
    - smalltalk
    - appreciation
    - confirmation
    - negation
    - help
    If it doesn't fit any of these, return "none".

    User: "{query}"
    Intent:
    """
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.1)
            intent = llm.invoke(prompt).content.strip().lower()
            if intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
                logger.info(f"[LLM Intent Fallback] → {intent}")
                return intent
        except Exception as e:
            logger.warning(f"[LLM Intent Detection Failed] {e}")
    
    return None
    


def is_likely_room_code(token: str) -> bool: # General, can be kept
    return bool(re.match(r"^\d+[a-z]([-_\s]?\d+[a-z])?$", token, re.IGNORECASE)) or \
           bool(re.match(r"^[A-Za-z]?\d{2,}[A-Za-z]?$", token)) # e.g. R101, 303B

def normalize_room_code(token: str) -> str:
    token = token.strip()
    token = re.sub(r"[-_\s]+", "", token) # Remove seperators
    token = re.sub(r"(\d)([a-z])", r"\1-\2", token)
    token = re.sub(r"([a-z])(\d)", r"\1-\2", token)
    return token.upper()

def normalize_query(query: str) -> str:
    """
    Enhanced query normalization with slang, abbreviations, and typo handling.
    """
    # 1. Basic cleaning
    query = query.strip()
    
    # 2. Slang and colloquial replacements
    slang_dict = {
        "pls": "please", "plz": "please", "thx": "thanks", "u": "you", "ur": "your",
        "r": "are", "btw": "by the way", "info": "information", "doc": "doctor", "dept": "department",
        "avail": "availability", "timng": "timing", "tmrw": "tomorrow", "tmr": "tomorrow",
        "w/": "with", "w/o": "without", "abt": "about", "bcoz": "because", "bcz": "because",
        "msg": "message", "num": "number", "ph": "phone", "mob": "mobile", "add": "address",
        "idk": "I don't know", "im": "I am", "dont": "don't", "cant": "can't", "wanna": "want to",
        "gonna": "going to", "gotta": "got to", "lemme": "let me", "kinda": "kind of", "sorta": "sort of",
        "outta": "out of", "lotta": "lot of", "ain't": "is not", "wanna": "want to", "ya": "you",
        "cuz": "because", "coz": "because", "nite": "night", "nite": "night", "luv": "love"
    }
    for slang, replacement in slang_dict.items():
        query = re.sub(rf"\b{slang}\b", replacement, query, flags=re.IGNORECASE)
    
    # 3. Abbreviations and contractions
    abbreviations = {
        "dr.": "doctor", "dr": "doctor",
        "dept.": "department", "dept": "department",
        "hosp.": "hospital", "hosp": "hospital",
        "appt.": "appointment", "appt": "appointment",
        "med.": "medical", "med": "medical",
        "info.": "information", "info": "information",
        "loc.": "location", "loc": "location",
        "num.": "number", "num": "number",
        "tel.": "telephone", "tel": "telephone",
        "addr.": "address", "addr": "address"
    }
    for abbr, full in abbreviations.items():
        query = re.sub(rf"\b{abbr}\b", full, query, flags=re.IGNORECASE)
    
    # 4. Standardize separators and special characters
    query = re.sub(r'[_/\\]', ' ', query)
    query = re.sub(r'[\s]+', ' ', query)
    
    # 5. Formatting instructions
    format_indicators = {
        r'\b(?:in|as|with|using)\s+bullet\s*points?\b': '[FORMAT:BULLETS]',
        r'\b(?:in|as|with|using)\s+(?:a\s+)?table\b': '[FORMAT:TABLE]',
        r'\b(?:step\s+by\s+step)\b': '[FORMAT:STEPS]',
        r'\b(?:in|as|with|using)\s+list\b': '[FORMAT:LIST]'
    }
    for pattern, replacement in format_indicators.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    # 6. Numbers and units
    query = re.sub(r'(\d+)\s*(rd|th|st|nd)\b', r'\1', query)
    
    # 7. Medical terms
    medical_terms = {
        r'\b(?:emergency\s*room|er)\b': 'emergency department',
        r'\b(?:operation\s*theater|ot)\b': 'operation theatre',
        r'\b(?:out\s*patient|opd)\b': 'outpatient department',
        r'\b(?:ultra\s*sound|usg)\b': 'ultrasound',
        r'\bicu\b': 'intensive care unit',
        r'\bccw\b': 'critical care ward',
        r'\bent\b': 'ear nose throat'
    }
    for pattern, replacement in medical_terms.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    # 8. Question patterns
    question_patterns = {
        r'\b(?:could|can)\s+you\s+(?:please\s+)?(?:tell|show|give)\s+me\b': 'what is',
        r'\bi\s+(?:want|need)\s+to\s+know\b': 'what is',
        r'\bdo\s+you\s+know\b': '',
        r'\bplease\s+(?:tell|show|give)\s+me\b': 'what is'
    }
    for pattern, replacement in question_patterns.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    # 9. Aggressive typo correction (optional, can be expanded)
    typo_dict = {
        "timng": "timing", "availibility": "availability", "departmnt": "department",
        "hospitl": "hospital", "docter": "doctor", "apointment": "appointment",
        "schedual": "schedule", "adres": "address", "contct": "contact"
    }
    for typo, correct in typo_dict.items():
        query = re.sub(rf"\b{typo}\b", correct, query, flags=re.IGNORECASE)
    
    return query.strip()

def preprocess_query(query: str, memory: ConversationMemory = None) -> tuple:
    """
    Comprehensive query preprocessing pipeline
    Returns: (processed_query, metadata)
    """
    metadata = {
        "original_query": query,
        "detected_language": None,
        "formatting": None,
        "query_type": None,
        "complexity": None
    }
    
    # 1. Extract formatting instructions
    formatting = extract_formatting_instructions(query)
    if formatting:
        metadata["formatting"] = formatting
        
    # 2. Language detection and translation
    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(query)
    translated_query, detected_input_lang = translator_manager.translate_to_english(cleaned_query_for_lang_detect, target_lang_code)
    metadata["detected_language"] = detected_input_lang
    
    # 3. Query normalization
    normalized_query = normalize_query(translated_query)
    
    # 4. Coreference resolution if memory available
    if memory and len(normalized_query.split()) < 7:
        processed_query = rewrite_query_with_memory(normalized_query, memory)
    else:
        processed_query = normalized_query
    
    # 5. Detect query complexity
    complexity = detect_query_complexity(processed_query)
    metadata["complexity"] = complexity
    
    # 6. Classify query type
    query_chars = classify_query_characteristics(processed_query)
    metadata["query_type"] = query_chars
    
    return processed_query, metadata

def classify_query_characteristics(query: str) -> dict:
    """
    Enhanced query classification with multiple characteristics
    """
    characteristics = {
        "response_length": "medium",  # default
        "expected_format": "paragraph",  # default
        "tone": "professional",  # default
        "priority": "normal",  # default
        "requires_context": False,
        "requires_clarification": False,
        "is_followup": False
    }
    
    # 1. Analyze query length and complexity
    word_count = len(query.split())
    if word_count <= 5:
        characteristics["response_length"] = "short"
    elif word_count >= 15:
        characteristics["response_length"] = "long"
    
    # 2. Detect format preferences
    if any(pattern in query.lower() for pattern in ["list", "bullet", "steps", "points"]):
        characteristics["expected_format"] = "list"
    elif "table" in query.lower():
        characteristics["expected_format"] = "table"
    elif any(pattern in query.lower() for pattern in ["explain", "elaborate", "details"]):
        characteristics["expected_format"] = "detailed"
    
    # 3. Analyze tone requirements
    if any(pattern in query.lower() for pattern in ["emergency", "urgent", "immediate"]):
        characteristics["tone"] = "urgent"
        characteristics["priority"] = "high"
    elif any(pattern in query.lower() for pattern in ["please", "kindly", "could you"]):
        characteristics["tone"] = "polite"
    
    # 4. Check for context dependency
    pronouns = ["it", "this", "that", "they", "them", "their", "these", "those"]
    if any(pronoun in query.lower().split() for pronoun in pronouns):
        characteristics["requires_context"] = True
        characteristics["is_followup"] = True
    
    # 5. Check for potential ambiguity
    if len(query.split()) <= 3 or "or" in query:
        characteristics["requires_clarification"] = True
    
    return characteristics

# Canonicalize entity value - uses SYNONYM_MAP, so it's fine.
def canonicalize_entity_value(entity_value):
    value_l = entity_value.lower().strip()
    for canonical, aliases in SYNONYM_MAP.items():
        all_forms = [canonical.lower()] + [a.lower() for a in aliases]
        if value_l in all_forms:
            return canonical
    # Fuzzy fallback using all canonical keys
    from difflib import get_close_matches
    all_canonicals = list(SYNONYM_MAP.keys())
    close = get_close_matches(value_l, all_canonicals, n=1, cutoff=0.85)
    if close:
        logger.info(f"[Fuzzy Canonicalization] '{entity_value}' → '{close[0]}'")
        return close[0]

    return entity_value  # No match found

def generate_clarification_suggestions(entities, memory):  # General logic, hospital-specific
    suggestions = []
    recent_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=2)

    # Doctor present but no department
    if entities.get("doctors") and not entities.get("departments"):
        for entity_info in recent_entities:
            if entity_info["type"] == "floors":
                suggestions.append(f"Could you specify which department Dr. {entities['doctors'][0]} belongs to?")
                break

    # Service present but no known location
    if entities.get("services") and not (entities.get("departments") or entities.get("rooms")):
        suggestions.append(f"Where is the {entities['services'][0]} service offered (e.g., department or room)?")

    # Department present, but doctor missing
    if entities.get("departments") and not entities.get("doctors"):
        dept = entities['departments'][0]
        suggestions.append(f"Which doctor in the {dept} department are you referring to?")

    # If department only, try to disambiguate intent
    if len(entities) == 1 and "departments" in entities:
        dept = entities['departments'][0]
        suggestions.append(f"Are you asking about a doctor, location, or services in the {dept} department?")

    # Room ambiguity
    if entities.get("rooms") and not entities.get("departments"):
        room = entities['rooms'][0]
        suggestions.append(f"Which department or floor is the room '{room}' part of?")

    # Fallback suggestions
    if not suggestions and (entities.get("rooms") or entities.get("departments")):
        suggestions.append("Can you provide more details or rephrase your query?")
    elif not suggestions:
        suggestions.extend([
            "Could you please provide more specific details?",
            "Can you rephrase your question?"
        ])

    return suggestions[:2]  # Return top 2

def classify_query_characteristics(query: str) -> dict:
    query_l = query.lower()
    word_count = len(query_l.split())

    # Default
    response_length = "short"

    # Heuristic: based on phrasing
    if any(phrase in query_l for phrase in ["explain in detail", "everything about", "detailed explanation"]):
        response_length = "long"
    elif any(phrase in query_l for phrase in ["list", "summarize", "overview of", "comparison of"]):
        response_length = "medium"
    elif any(phrase in query_l for phrase in ["tell me more", "a bit more detail", "summary of", "overview", "more info"]):
        response_length = "medium"

    # Heuristic: based on length of query itself
    if word_count > 15:
        response_length = "long"
    elif word_count > 7 and response_length == "short":
        response_length = "medium"

    return {"response_length": response_length}


def detect_answer_style_and_tone(query: str) -> tuple:
    query_l = query.lower()
    style = "paragraph"
    tone = "professional_and_helpful"

    # Style detection
    if "in a table" in query_l or "tabular format" in query_l:
        style = "table"
    elif any(phrase in query_l for phrase in ["bullet points", "list them", "pointwise", "in bullets", "quick list"]):
        style = "bullet_list"
    elif "quick answer" in query_l or "short answer" in query_l:
        style = "bullet_list"

    # Tone detection
    if any(word in query_l for word in ["casual", "friendly", "informal", "chill"]):
        tone = "friendly_and_casual"
    if any(word in query_l for word in ["formal", "official", "professional tone", "official statement", "be polite"]):
        tone = "formal_and_precise"
    
    return style, tone

def rewrite_query_with_memory(query, memory: ConversationMemory):
    """Rewrite the query using conversation context and memory."""
    original_query = query.strip()
    rewritten_query = original_query
    query_lower_normalized = normalize_query(original_query.lower())

    # Get recent context
    context_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
    recent_history = memory.history[-3:] if memory.history else []
    
    # Get the most salient topic with enhanced priority
    salient_topic_entity_value = memory.get_last_entity_by_priority(
        type_priority=["doctors", "departments", "services", "rooms", "hospitals", "buildings"]
    )
    
    # Track entity types for better context understanding
    salient_topic_type = None
    if salient_topic_entity_value:
        for entity_type in ["doctors", "departments", "services", "rooms", "hospitals", "buildings"]:
            if entity_type in memory.last_entity_by_type and memory.last_entity_by_type[entity_type]["value"] == salient_topic_entity_value:
                salient_topic_type = entity_type
                break

    # Enhanced pronoun and reference resolution
    pronouns = {
        "personal": ["he", "she", "it", "they", "them", "his", "her", "their", "theirs"],
        "demonstrative": ["this", "that", "these", "those"],
        "relative": ["which", "who", "whom", "whose"],
        "possessive": ["its", "their", "theirs", "his", "hers"]
    }
    
    # Check for pronouns and references that need resolution
    query_tokens = query_lower_normalized.split()
    has_pronoun = any(token in [p for plist in pronouns.values() for p in plist] for token in query_tokens)
    has_reference = any(ref in query_lower_normalized for ref in ["the", "that", "this", "these", "those"])
    
    # Enhanced contextual keywords that might need entity resolution
    contextual_keywords = {
        "location": ["where", "located", "find", "location", "place", "address"],
        "contact": ["contact", "phone", "email", "reach", "available"],
        "time": ["when", "timing", "schedule", "hours", "available"],
        "service": ["service", "provide", "offer", "available", "facility"],
        "person": ["doctor", "staff", "specialist", "consultant", "person"],
    }

    # Check if query contains contextual keywords
    context_type = None
    for ctype, keywords in contextual_keywords.items():
        if any(keyword in query_lower_normalized for keyword in keywords):
            context_type = ctype
            break

    # Determine if we need to rewrite based on various factors
    needs_rewrite = (
        has_pronoun or 
        has_reference or 
        (context_type and salient_topic_entity_value) or
        len(query_tokens) <= 3  # Short queries might need context
    )

    if needs_rewrite:
        # Build context-aware query
        if salient_topic_entity_value:
            # Handle different types of contextual queries
            if context_type == "location" and salient_topic_type:
                rewritten_query = f"Where is {salient_topic_entity_value} {salient_topic_type} located?"
            elif context_type == "contact" and salient_topic_type:
                rewritten_query = f"What are the contact details for {salient_topic_entity_value} {salient_topic_type}?"
            elif context_type == "time" and salient_topic_type:
                rewritten_query = f"What are the timings or schedule for {salient_topic_entity_value} {salient_topic_type}?"
            elif context_type == "service" and salient_topic_type:
                rewritten_query = f"What services are available at {salient_topic_entity_value} {salient_topic_type}?"
            elif context_type == "person" and salient_topic_type:
                rewritten_query = f"Who is {salient_topic_entity_value} in the {salient_topic_type} department?"
            else:
                # General reference resolution
                for pronoun_type, pronoun_list in pronouns.items():
                    for pronoun in pronoun_list:
                        if f" {pronoun} " in f" {query_lower_normalized} ":
                            rewritten_query = query.replace(
                                pronoun, 
                                f"{salient_topic_entity_value} {salient_topic_type if salient_topic_type else ''}"
                            ).strip()
                            break

        # Handle demonstrative references without clear entity
        if has_reference and not salient_topic_entity_value and recent_history:
            last_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=1)
            if last_entities:
                # Get the most recent entity
                for entity_type, entities in last_entities.items():
                    if entities:
                        latest_entity = entities[0]["value"]
                        rewritten_query = query.replace("this", latest_entity).replace("that", latest_entity)
                        break

    # Log the rewriting process
    if rewritten_query != original_query:
        logger.info(f"[Query Rewrite] Original: '{original_query}' → Rewritten: '{rewritten_query}'")
        logger.info(f"[Rewrite Context] Topic: {salient_topic_entity_value} ({salient_topic_type}), Context Type: {context_type}")
    
    return rewritten_query


def handle_small_talk(user_query: str, memory, user_id: str, detected_input_lang: str = "en", convo_intent: str = None)-> dict:
    import random
    query_corrected = correct_spelling(user_query)
    query_cleaned = collapse_repeated_letters(query_corrected.lower().strip())

    has_prior_convo = memory and len(memory.history) > 1
    recent_topics = []
    recent_entities = {}
    
    # Get context from recent conversation
    if has_prior_convo:
        recent_topics = [turn["user"] for turn in memory.history[-3:]]  # Last 3 turns
        recent_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
    
    # Personalize greeting based on time of day and conversation history
    current_hour = datetime.now().hour
    time_of_day = (
        "morning" if 5 <= current_hour < 12
        else "afternoon" if 12 <= current_hour < 17
        else "evening" if 17 <= current_hour < 22
        else "night"
    )
    
    # Build personalization context
    context_parts = []
    if has_prior_convo:
        context_parts.append("Welcome back! 😊")
    else:
        context_parts.append(f"Good {time_of_day}! 👋")
    
    # Add relevant context from recent conversation
    if recent_entities:
        for entity_type, entities in recent_entities.items():
            if entity_type in ["doctors", "departments", "services"]:
                latest_entity = entities[0]["value"]
                if convo_intent == "greeting":
                    context_parts.append(f"I see you were asking about {latest_entity}.")
                    break

    personalization_prefix = " ".join(context_parts)

    # Enhance prompt with conversation context
    convo_prompt = f"""
You are a warm, friendly assistant working at AIIMS Jammu. 
The user is speaking casually — respond with kindness, sometimes humor, and use emojis naturally.
Avoid repeating yourself. Always vary your response. 
Add a touch of human-like warmth and keep it short and refreshing.

Recent conversation context:
- Time of day: {time_of_day}
- Has prior conversation: {"Yes" if has_prior_convo else "No"}
- Recent topics discussed: {", ".join(recent_topics) if recent_topics else "None"}
- Conversation intent: {convo_intent if convo_intent else "general chat"}

{personalization_prefix}
User: {query_cleaned}
Assistant:"""

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="gemma-2-9b-it", temperature=0.85)
        response = llm.invoke(convo_prompt).content.strip()
        
        # Applied max length guard
        if len(response) > 500:
            response = response[:500] + "..."

        # Emoji variety based on conversation intent
        intent_emoji_map = {
            "greeting": ["👋", "😊", "🌟", "🤗"],
            "exit": ["👋", "😊", "✨", "👍"],
            "appreciation": ["🙏", "💖", "✨", "🌟"],
            "smalltalk": ["😊", "😄", "🌟", "🙂"],
            "confirmation": ["👍", "✅", "💯", "🎯"],
            "negation": ["🤔", "💭", "❓", "🔄"],
            "help": ["💡", "🤝", "✨", "💪"]
        }
        
        emoji_pool = intent_emoji_map.get(convo_intent, ["😊", "👋", "😄", "🌟", "🙂", "🙌", "👍", "🤗"])
        if not any(emoji in response for emoji in emoji_pool):
            response += " " + random.choice(emoji_pool)

    except Exception as e:
        logger.error(f"Error in small talk handler: {e}")
        # Enhanced fallback responses with context
        fallback_responses = [
            f"Hey there! 👋 How can I assist you at AIIMS Jammu this {time_of_day}?",
            "Hi! 😊 I'm here to help with information about our departments, doctors, and services.",
            "Welcome! 🌟 Feel free to ask me anything about AIIMS Jammu."
        ]
        if has_prior_convo:
            fallback_responses.extend([
                "Welcome back! 🤗 How else can I help you today?",
                "Great to see you again! 👋 What information do you need?"
            ])
        response = random.choice(fallback_responses)

    # Enhanced repetition check with variation
    if memory and memory.history:
        last_reply = memory.history[-1]["assistant"]
        if response.strip() == last_reply.strip():
            variations = [
                "😊 Is there anything specific you'd like to know about AIIMS Jammu?",
                "🤗 How else can I assist you today?",
                "👋 Feel free to ask about our departments or services!",
                "💡 I'm here to help with any questions you might have.",
                "✨ What would you like to know more about?"
            ]
            response += " " + random.choice(variations)

    # Save interaction context
    context_data = {
        "time_of_day": time_of_day,
        "conversation_intent": convo_intent,
        "has_prior_conversation": has_prior_convo
    }
    
    # Save small talk to memory with context
    memory.add_turn(query_cleaned, response, extracted_entities_map={"conversation_context": context_data})
    memory.add_important_context("last_interaction_time", datetime.now().isoformat())
    user_memory_store.save(user_id, memory)

    # Optional: Translate back to user's language (if not English)
    if detected_input_lang and detected_input_lang != "en":
        try:
            response_translated = GoogleTranslator(source="en", target=detected_input_lang).translate(response)
            logger.info(f"[Small Talk] Translated to {detected_input_lang}")
        except Exception as e:
            logger.warning(f"[Small Talk] Translation failed: {e}")
            response_translated = response
    else:
        response_translated = response

    return {
        "answer": response_translated,
        "debug_info": {
            "source_answer": response,
            "language": detected_input_lang,
            "personalized": has_prior_convo,
            "intent": convo_intent,
            "context": context_data
        }
    }


def chat(user_query: str, user_id: str):
    request_start_time = datetime.now()
    logger.info(f"--- New Chat Request (Hospital) --- User ID: {user_id} | Query: '{user_query}'")

    # Get existing memory or new one from your custom memory store
    conv_memory = user_memory_store.get(user_id)

    original_user_query = user_query.strip()
    # user_query = correct_spelling(user_query)

    query_lower_raw = original_user_query.lower()
    # Early check for conversational intents that might not need full processing pipeline
    # Detect conversational intent BEFORE translation or complex normalization
    convo_intent = detect_conversational_intent(original_user_query) 
    hospital_entity_keywords = [
        # People & Roles
        "doctor", "dr", "nurse", "staff", "physician", "consultant", "specialist", "technician", "receptionist", "attendant",

        # Facilities & Infrastructure
        "hospital", "aiims", "building", "floor", "room", "ward", "icu", "opd", "ot", "emergency", 
        "pharmacy", "reception", "entrance", "exit", "waiting area", "lift", "staircase", "toilet", "lab", 
        "laboratory", "radiology", "x-ray", "mri", "ct", "ultrasound", "scan room", "parking", "canteen",

        # Departments / Specialties
        "department", "cardiology", "neurology", "oncology", "orthopedics", "pediatrics", "radiology",
        "dermatology", "gynecology", "ophthalmology", "ent", "surgery", "psychiatry", "anesthesiology", 
        "emergency medicine", "nephrology", "urology", "surgical gastroenterology",

        # Services & Appointments
        "service", "clinic", "test", "scan", "diagnosis", "appointment", "registration", "admission", 
        "discharge", "treatment", "procedure", "report", "vaccination", "blood test", "dialysis", "checkup",

        # Navigation & Location
        "location", "find", "where", "nearby", "direction", "navigate", "way", "reach", "path", "map", "route",

        # Contact & Timings
        "contact", "phone", "email", "connect", "availability", "schedule", "hours", "timings", "operatinghours"
        "time", "day", "date", "open", "closed",

        # Devices & Equipment
        "ventilator", "oxygen", "ecg", "bed", "monitor", "stretcher", "ambulance", "wheelchair", "equipment",

        # Accessibility
        "accessible", "disabled", "wheelchair access", "braille", "signage", "elevator",

        # Metadata
        "id", "code", "number", "feedback", "helpdesk", "volunteer"
    ]
    
    # If conversational intent detected AND query doesn't seem to contain substantial entity keywords, handle as small talk
    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(keyword in query_lower_raw for keyword in hospital_entity_keywords):
             # Pass original query to small talk handler to preserve nuance
             return handle_small_talk(original_user_query, conv_memory, user_id, detected_input_lang)


    # Proceed with full pipeline for substantive queries or mixed conversational/substantive queries
    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
    translated_query, detected_input_lang = translator_manager.translate_to_english(cleaned_query_for_lang_detect)
    processed_query = (
        rewrite_query_with_memory(translated_query, conv_memory) 
        if len(translated_query.split()) < 7 and not any(translated_query.lower().startswith(q_word) for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"]) 
        else translated_query
    )

    # NLU processing on the potentially rewritten and normalized query
    extracted_query_entities = nlu_processor.extract_entities_with_fallback(processed_query)
    # Enhance with spaCy NER HERE
    spacy_entities = extract_spacy_ner_entities(processed_query)
    extracted_query_entities = merge_entities(extracted_query_entities, spacy_entities)
    # Enhance with PhraseMatcher-based entities from spaCy
    spacy_matched_entities = extract_entities_spacy(processed_query)
    extracted_query_entities = merge_entities(extracted_query_entities, spacy_matched_entities)

    # Extract formatting instructions separately from entities
    formatting_instructions = extract_formatting_instructions(processed_query)
    if formatting_instructions.get("format"):
        logger.info(f"[Formatting] Detected format instruction: {formatting_instructions['format']}")
        # Override answer style if formatting is specified
        answer_style = formatting_instructions["format"]

    # === Canonicalize all entity values using SYNONYM_MAP ===
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

    # === Tag Matching Boost (from hospital metadata tags) ===
    known_tags = set(data_loader.get_all_metadata_tags())  # Cache globally if performance matters

    for key, val in extracted_query_entities.items():
        if isinstance(val, str) and val.lower() in known_tags:
            logger.info(f"[Tag Match] Entity '{val}' matched known metadata tag")
            extracted_query_entities[f"matched_tag__{key}"] = val.lower()
        elif isinstance(val, list):
            for subval in val:
                if isinstance(subval, str) and subval.lower() in known_tags:
                    logger.info(f"[Tag Match] List Entity '{subval}' matched known tag")
                    extracted_query_entities.setdefault(f"matched_tag__{key}", []).append(subval.lower())

    # === Fallback: Enrich sparse entities using memory if current query lacks them ===
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

    # === Intent Detection ===
    task_type, intent_confidence = nlu_processor.classify_intent_with_fallback(processed_query)
    convo_intent = detect_conversational_intent(processed_query)

    # === Smart Error Handling & Clarification ===
    error_response = None
    
    # 1. Check intent confidence
    if intent_confidence < 0.6:
        error_response = {
            "message": "I'm not quite sure what you're asking. Could you be more specific?",
            "suggestions": [
                "Are you looking for location information?",
                "Do you need contact details?",
                "Would you like to book an appointment?"
            ]
        }
    
    # 2. Check for empty or ambiguous entities
    elif not any(extracted_query_entities.values()):
        error_response = {
            "message": "Could you provide more specific details?",
            "suggestions": [
                "Mention specific department names",
                "Use doctor names if you know them",
                "Specify what kind of information you need"
            ]
        }
    
    # 3. Check for very short or ambiguous queries
    elif len(processed_query.split()) <= 3:
        error_response = {
            "message": "Your query is a bit short. Could you provide more details?",
            "suggestions": [
                "What specific information are you looking for?",
                "Which department are you interested in?",
                "What type of service do you need?"
            ]
        }
    
    if error_response:
        # Add context-aware suggestions if available
        if conv_memory and conv_memory.history:
            recent_entities = conv_memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
            if recent_entities:
                for entity_type, entities in recent_entities.items():
                    if entities:
                        error_response["suggestions"].append(
                            f"Ask about {entity_type}: {', '.join(str(e) for e in entities[:2])}"
                        )
        
        # Save interaction in memory
        conv_memory.add_turn(original_user_query, error_response["message"], extracted_query_entities)
        user_memory_store.save(user_id, conv_memory)
        
        return {
            "answer": error_response["message"],
            "suggestions": error_response["suggestions"][:3],
            "requires_clarification": True
        }

    query_lower_raw = original_user_query.lower()


    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(word in query_lower_raw for word in hospital_entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, user_id, detected_input_lang)
        else:
            # Treat as mixed: greeting + actual question → combine both
            greeting_resp = handle_small_talk(original_user_query, conv_memory, user_id, detected_input_lang)["answer"]
            # We'll append this to final LLM response later

    # Store original query in memory (with empty assistant response initially)
    conv_memory.add_turn(original_user_query, "", extracted_query_entities)
    user_memory_store.save(user_id, conv_memory)  # Save memory after updating it
    logger.info(f"Detected task type (hospital): {task_type}")


    if task_type == "out_of_scope":
        # Use hospital-specific OOS message
        out_of_scope_response = "I am an assistant for aimms jammu hospital and can only answer questions related to its facilities, departments, doctors, services, and appointments. How can I help you with that?"
        conv_memory.history[-1]["assistant"] = out_of_scope_response # Update last turn's assistant response
        user_memory_store.save(user_id, conv_memory)
        try:
            if target_lang_code and target_lang_code != "en": 
                out_of_scope_response = GoogleTranslator(source="en", target=target_lang_code).translate(out_of_scope_response)
            elif detected_input_lang != "en": 
                out_of_scope_response = GoogleTranslator(source="en", target=detected_input_lang).translate(out_of_scope_response)
        except Exception as e:
            logger.warning(f"[Translation-Fallback] OOS message translation failed: {e}")
        return {"answer": out_of_scope_response, "debug_info": {"task_type": task_type, "reason": "Query did not match any hospital-related intents"}}

    if not GROQ_API_KEY: 
        logger.critical("Groq API key not configured."); 
        return {"answer": "Error: Chat service temporarily unavailable."}
    query_chars = classify_query_characteristics(processed_query)
    response_length_hint = query_chars.get("response_length", "short") # Default to short
    answer_style, answer_tone = detect_answer_style_and_tone(processed_query)
    logger.info(f"Response hints (hospital): length={response_length_hint}, style={answer_style}, tone={answer_tone}")

    # Step 1: Expand user query to multiple variants
    query_variants = expand_query_with_synonyms(processed_query)
    query_variants = query_variants[:3]  # Limit to top 3 variants for performance
    logger.info(f"[Synonym Expansion] Variants generated: {query_variants}")

    # Step 2: Retrieve docs for each variant and merge results
    all_retrieved_docs = []
    for variant in query_variants:
        retrieved = hybrid_retriever(variant, k_simple=6, k_normal=10, k_complex=15)
        if retrieved:
            all_retrieved_docs.extend(retrieved)

    # Step 3: Deduplicate by doc ID or content
    unique_docs = {}
    for doc in all_retrieved_docs:
        doc_id = doc.metadata.get("source_doc_id") or hash(doc.page_content)
        unique_docs[doc_id] = doc
    retrieved_docs = list(unique_docs.values())
    logger.info(f"[Synonym Retrieval] Total unique docs after expansion: {len(retrieved_docs)}")

    if not retrieved_docs:
        logger.warning(f"No documents retrieved for hospital query: {processed_query}")

        # Identify primary entity for fallback message
        primary_entity = None
        for ent_type in ["departments", "services", "doctors", "rooms", "buildings"]:
            vals = extracted_query_entities.get(ent_type)
            if vals:
                primary_entity = vals[0]
                break

        # Determine fallback template based on task type
        fallback_msg = ""
        suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
        bullet_suggestions = "\n• " + "\n• ".join(suggestions) if suggestions else ""

        # Task-specific fallback messages
        if task_type in {"location", "location_specific", "location_general"}:
            fallback_msg = f"Sorry, I couldn't find the **location details** for **{primary_entity or 'the requested area'}**."
        elif task_type == "contact_info":
            fallback_msg = f"I'm unable to locate **contact information** for **{primary_entity or 'the specified entity'}**."
        elif task_type in {"operating_hours", "doctor_availability"}:
            fallback_msg = f"I couldn't find **availability or timings** for **{primary_entity or 'the requested department or doctor'}**."
        elif task_type in {"service_info", "department_info", "general_information"}:
            fallback_msg = f"Unfortunately, I couldn't retrieve detailed **information** about **{primary_entity or 'the requested topic'}**."
        else:
            fallback_msg = "I couldn’t find relevant information in the current data."

        full_msg = (
            f"{fallback_msg}\n\n"
            f"You could try the following:{bullet_suggestions}" if suggestions else fallback_msg
        )

        # Update memory
        if conv_memory.history:
            conv_memory.history[-1]["assistant"] = full_msg
            user_memory_store.save(user_id, conv_memory)

        # Translate fallback response
        try:
            if target_lang_code and target_lang_code != "en":
                full_msg = GoogleTranslator(source="en", target=target_lang_code).translate(full_msg)
            elif detected_input_lang != "en":
                full_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(full_msg)
        except Exception as e:
            logger.warning(f"[Translation-Fallback] Clarification translation failed: {e}")

        return {
            "answer": full_msg,
            "related_queries": suggestions,
            "requires_clarification": True
        }


    fast_reranked_docs = rerank_documents_fast(processed_query, retrieved_docs, top_k=15) # Increase fast reranker top_k
    final_docs_for_llm = rerank_documents_strong(processed_query, fast_reranked_docs, top_k=3) # Increase strong reranker top_k

    # Combine doctors + persons entity pools
    doctor_candidates = extracted_query_entities.get("doctors", []) + extracted_query_entities.get("persons", [])
    doctor_candidates = [d for d in doctor_candidates if len(d.split()) >= 2]  # only full names
    query_doctor_name = doctor_candidates[0] if doctor_candidates else extract_doctor_name(processed_query)

    # Always check for doctor matches if a name is present
    if query_doctor_name:
        response = get_doctor_by_name(query_doctor_name, final_docs_for_llm)
        if response:
            logger.info(f"[Doctor Match] Structured match found for: {query_doctor_name}")
            return {"answer": response}
        else:
            logger.info(f"[Doctor Match] No match found for: {query_doctor_name}")

    # Entity grounding check
    entity_terms_to_check = set()
    for ent_list in extracted_query_entities.values():
        for val in ent_list:
            val_clean = val.lower().strip()
            # Filter out junk tokens like '##p' or anything too short
            if val_clean and len(val_clean) > 1 and not val_clean.startswith("##"):
                entity_terms_to_check.add(val_clean)

    logger.info(f"[Entity Grounding] Checking for terms in docs: {entity_terms_to_check}")

    missing_entities = []
    entity_grounding_info = {}
    for term in entity_terms_to_check:
        score, doc, field = ground_entity_to_docs(term, final_docs_for_llm)
        entity_grounding_info[term] = {
            "score": round(score, 3),
            "field": field,
            "doc_id": doc.metadata.get("source_doc_id") if doc else None
        }
        if score < ENTITY_GROUNDING_THRESHOLD:
            missing_entities.append(term)

    logger.info(f"[Entity Grounding Scores] {entity_grounding_info}")

    # If key terms are not found in the context, ask for clarification
    task_type = task_type if "task_type" in locals() else "general_information"
    if missing_entities and task_type != "general_information":
        logger.warning(f"[Missing Context] Could not find these terms in retrieved docs: {missing_list}")
        missing_list = ", ".join(missing_entities)

        # Try to suggest something useful based on entity types
        primary_entity = missing_entities[0] if missing_entities else "the requested topic"
        clarification_msg = f"Unfortunately, I couldn't find relevant information about **{primary_entity}** in the current hospital data."

        # Append task-specific intent to clarification message
        if task_type in {"location", "location_specific"}:
            clarification_msg += " The location details may not be available yet."
        elif task_type == "contact_info":
            clarification_msg += " It seems contact details are missing."
        elif task_type == "doctor_availability":
            clarification_msg += " Their schedule might not be available right now."
        elif task_type in {"department_info", "service_info"}:
            clarification_msg += " This department or service might not be documented yet."

        # Try generating helpful follow-up suggestions
        clarification_suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
        if clarification_suggestions:
            clarification_msg += "\n\nYou could try:\n" + "\n".join(f"• {s}" for s in clarification_suggestions[:4])

        # Update memory
        if conv_memory.history:
            conv_memory.history[-1]["assistant"] = clarification_msg
            user_memory_store.save(user_id, conv_memory)

        # Translate message if needed
        try:
            if target_lang_code and target_lang_code != "en":
                clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
            elif detected_input_lang != "en":
                clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)
        except Exception as e:
            logger.warning(f"[Translation-Fallback] Clarification translation failed: {e}")

        return {
            "answer": clarification_msg,
            "related_queries": clarification_suggestions[:3] if clarification_suggestions else [],
            "missing_entities": missing_entities,
            "requires_clarification": True
        }

            

    # Force-inject top BM25 document if different and seems relevant
    top_bm25_docs_for_injection = bm25_retriever_func(processed_query, k=1) # Get just the top 1
    if top_bm25_docs_for_injection:
        top_bm25_doc = top_bm25_docs_for_injection[0]
        if all(top_bm25_doc.page_content.strip() != doc.page_content.strip() for doc in final_docs_for_llm):
            final_docs_for_llm.append(top_bm25_doc)
            logger.info("Injected top BM25 doc into LLM context.")

    logger.info(f"Final {len(final_docs_for_llm)} documents selected for LLM context (hospital).")
    if not final_docs_for_llm and retrieved_docs: final_docs_for_llm = retrieved_docs[:3]; logger.warning("Reranking resulted in zero documents. Using top 3 from initial hybrid retrieval for LLM.")

    context_parts = []
    for i, doc in enumerate(final_docs_for_llm):
        doc_text = f"Source Document {i+1}:\n{doc.page_content}\n"
        # Hospital specific metadata to show in context string for LLM
        meta_info = {
            "Hospital": doc.metadata.get("hospital_name"),
            "Building": doc.metadata.get("building_name"),
            "Floor": doc.metadata.get("floor"),
            "Room Name": doc.metadata.get("room_name"),
            "Room Number": doc.metadata.get("room_number"),
            "Associated Depts": ", ".join(ensure_list(doc.metadata.get("associated_departments", []))[:2]), # Show first 2
            "Associated Doctors": ", ".join(ensure_list(doc.metadata.get("associated_doctors", []))[:2]), # Show first 2
            "Key Services": (", ".join(ensure_list(doc.metadata.get("services_directly_offered", []))[:2]) or
                             ", ".join(ensure_list(doc.metadata.get("department_related_services", []))[:2])),
            "Doc ID": doc.metadata.get("source_doc_id")
        }
        filtered_meta_info = {k: v for k, v in meta_info.items() if v is not None and v != ""} # Filter out empty/None/NA
        if filtered_meta_info:
            doc_text += "Key Metadata: " + "; ".join([f"{k}: {v}" for k, v in filtered_meta_info.items()])
        context_parts.append(doc_text)
    extracted_context_str = "\n\n---\n\n".join(context_parts)

    prompt_intro = f"You are a highly advanced, intelligent, and conversational AI assistant for AIMMS JAMMU Building. Your primary goal is to provide accurate, concise, and relevant information based ONLY on the 'Extracted Context' provided. If the context is insufficient or irrelevant, clearly state that you cannot answer or need more information. Do NOT invent information or use external knowledge."
    
    task_instructions = "" # Default empty
    if task_type in ["location", "location_specific", "location_general"]:
        task_instructions = (
            "When answering location-based queries, always provide clear and complete location details based ONLY on the Extracted Context. "
            "Include the hospital name, building name, zone/wing, floor number, and room number or name if present in the context. "
            "Avoid vague statements like 'located at AIIMS Jammu' unless that's all the context provides. "
            "If nearby landmarks or access points (like lifts, stairs, or entrances) are mentioned, include them too. "
            "Be precise, structured, and helpful."
        )
    elif task_type == "contact_info":
        task_instructions = "Extract and provide specific contact details like email, phone numbers, or website URLs for the queried entity (hospital, department, doctor) from the context. If multiple contacts exist, list them clearly."
    elif task_type == "operating_hours" or task_type == "doctor_availability":
        task_instructions = "Clearly state the operating hours, availability, days of the week, start, and end times as found in the context for the queried entity (e.g., OPD, doctor, service)."
    elif task_type in ["explanation", "general_information", "department_info", "service_info"]:
        task_instructions = "Provide a comprehensive explanation or description based on the context. If the context has a summary for a room or service, use it but elaborate with other details if available. For departments or services, describe what they are or what they offer based on context."
    elif task_type in ["listing_all", "listing_specific"]:
        task_instructions = "List all relevant items (e.g., doctors in a department, services offered, rooms on a floor) based on the query and context. Use bullet points if appropriate for clarity."
    elif task_type == "booking_info": # For appointments
        task_instructions = "Provide details on how to book an appointment or access a service, including method, contact for booking, or relevant URLs if found in the context. Mention if approval is required."
    elif task_type == "comparison":
        task_instructions = "Compare the relevant entities (e.g., doctors, services, treatments) based on the information available in the context, highlighting differences and similarities in aspects like specialty, availability, or features."


    prompt_template_str = f"""{prompt_intro}

Strict Rules:
1. Base answers ONLY on 'Extracted Context'. If the information is not in the context, state that clearly (e.g., "Based on the provided information, I cannot answer that," or "The context does not contain details about X."). Do not use knowledge beyond this context. If multiple possible answers exist in the context, summarize them clearly. If context is insufficient, say so politely.
2. If the Extracted Context is empty or clearly irrelevant to the query, state that you lack the necessary information to answer.
3. Consider 'Past Conversation History' for resolving ambiguities (like "his email" referring to a previously discussed doctor) but prioritize the current query and the 'Extracted Context' as the source of truth for the answer.
4. If the query is ambiguous despite context and history, you can ask ONE brief clarifying question.
5. Be conversational, empathetic, and helpful, adapting to a hospital setting.
6. {task_instructions}
7. If asked about medical advice, conditions, or treatments, state that you are an AI assistant and cannot provide medical advice. Suggest consulting with a healthcare professional. However, if the query is about *information available in the context* regarding a service or procedure (e.g., "what does the context say about X-ray procedure?"), then answer based on the context.
8. When possible, return structured answers:
   - Use **bullet points** for lists (e.g., multiple doctors, rooms, departments).
   - Use **labels** (e.g., Room Number: 301, Department: Radiology) to format details clearly.
   - For comparisons or listings, use a **table format** if relevant fields (name, location, contact, etc.) are available.
   - Avoid vague phrases like "at AIIMS Jammu" if room name, floor, and building info are present — include those explicitly.
   - If the answer refers to a specific person or entity mentioned earlier, restate the name for clarity (e.g., "Dr. Aymen Masood is located in…").

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

Answer (provide only the answer, no preamble like "Here is the answer:") :
"""
    chat_history_for_prompt = conv_memory.get_contextual_history_text(num_turns=4) # Slightly less history for prompt
    llm_input_data = {
        "input": processed_query, 
        "context": extracted_context_str, 
        "history": chat_history_for_prompt,
        "task_type": task_type,
        "answer_style": answer_style,
        "answer_tone": answer_tone,
        "response_length_hint": response_length_hint
    }

    # LLM model selection based on complexity/task for hospital queries
    if response_length_hint == "long" or task_type in ["explanation", "comparison", "listing_all", "List"] or "complex" in detect_query_complexity(processed_query) :
        groq_llm_model_name, temperature_val = "deepseek-r1-distill-llama-70b", 0.4 # Slightly lower temp for factual long
    elif task_type in ["contact_info", "location", "doctor_availability"] and response_length_hint == "short":
        groq_llm_model_name, temperature_val = "llama3-8b-8192", 0.15 # Precise for short facts
    else: 
        groq_llm_model_name, temperature_val = "deepseek-r1-distill-llama-70b", 0.25 # Default to more capable model with moderate temp
    logger.info(f"[LLM Model Selector] Task: {task_type}, Length: {response_length_hint} → Model: {groq_llm_model_name}, Temp: {temperature_val}")
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model=groq_llm_model_name, temperature=temperature_val)
    prompt = PromptTemplate.from_template(prompt_template_str)
    runnable_chain = prompt | llm
    final_response_text = "Error: Could not generate a response for your hospital query." # Default error
    try:
        ai_message = runnable_chain.invoke(llm_input_data)
        final_response_text = ai_message.content
        logger.info(f"LLM Raw Response Snippet (hospital): {final_response_text[:250]}...")
    except Exception as e:
        logger.error(f"Error invoking RAG chain with Groq (hospital): {e}")
        final_response_text = "I apologize, but I encountered an issue while processing your request. The context might have been too large."

    conv_memory.history[-1]["assistant"] = final_response_text
    # Prepend greeting if this was a mixed greeting+query
    if convo_intent in {"greeting", "smalltalk"} and 'greeting_resp' in locals():
        final_response_text = f"{greeting_resp}\n\n{final_response_text}"

    user_memory_store.save(user_id, conv_memory) # Save updated memory to Redis

    # Store conversation summary into FAISS memory
    store_summary_to_faiss(user_id, conv_memory)

    # Translate response if needed
    try:
        if target_lang_code and target_lang_code != "en": 
            final_response_text = translator_manager.translate_from_english(final_response_text, target_lang_code)
            logger.info(f"Translated response to {target_lang_code}.")
        elif detected_input_lang != "en" and detected_input_lang is not None: # check detected_input_lang is not None
            final_response_text = translator_manager.translate_from_english(final_response_text, detected_input_lang)
            logger.info(f"Translated response back to input language {detected_input_lang}.")
    except Exception as e: 
        logger.warning(f"Failed to translate final response: {e}")
    
    processing_time = (datetime.now() - request_start_time).total_seconds()
    logger.info(f"--- Chat Request Completed (Hospital) --- Time: {processing_time:.2f}s")
    debug_info = {
        "user_id": user_id,
        "original_query": original_user_query,
        "processed_query": processed_query,
        "query_variants_for_retrieval": query_variants,  # from synonym expansion
        "detected_input_language": detected_input_lang,
        "target_response_language": target_lang_code,
        "detected_task_type": task_type,
        "conversational_intent": convo_intent,
        "extracted_entities": extracted_query_entities,
        "entity_grounding_check": entity_grounding_info,
        "missing_entities_in_docs": missing_entities,
        "retrieved_docs_count_initial": len(retrieved_docs) if retrieved_docs else 0,
        "retrieved_docs_count_final_llm": len(final_docs_for_llm) if final_docs_for_llm else 0,
        "final_doc_ids_for_llm": [doc.metadata.get("source_doc_id","Unknown") for doc in final_docs_for_llm] if final_docs_for_llm else [],
        "llm_model_used": groq_llm_model_name,
        "response_length_hint": response_length_hint,
        "answer_style": answer_style,
        "answer_tone": answer_tone,
        "query_complexity": classify_query_characteristics(processed_query),
        "processing_time_seconds": round(processing_time, 2),
        "timestamp_utc": datetime.utcnow().isoformat()
    }
    return {"answer": final_response_text, "debug_info": debug_info}


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("indextag.html", {"request": request})


@app.post("/refresh_data")
async def refresh_data_endpoint():
    logger.info("Hospital data refresh request received.")
    global data_loader
    data_loader = HospitalDataLoader()
    refresh_faiss_and_bm25()
    build_spacy_phrase_matcher()
    logger.info("Hospital data and retrieval models refreshed successfully.")
    return {"message": "Hospital data and retrieval models refreshed successfully."}


@app.get("/api/metadata-tags")
async def get_metadata_tags():
    tags = data_loader.get_all_metadata_tags()
    return {"tags": tags}

@app.post("/chat", tags=["Chat"], summary="Hospital chatbot endpoint")

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/tag-counts")
async def get_metadata_tag_counts():
    return {"tag_counts": data_loader.get_metadata_tag_counts()}


class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: Request, chat_data: ChatInput, x_user_id: str = Header(None)):
    user_message = chat_data.message.strip()
    if not user_message:
        return JSONResponse(status_code=400, content={"error": "No message provided"})

    user_id = x_user_id or request.client.host
    response = chat(user_message, user_id)
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    # Consider using a different port if the RNI version might also run
    uvicorn.run("main21:app", host="0.0.0.0", port=5001, reload=True)