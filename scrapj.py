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
from datetime import datetime
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
from langchain.memory import ConversationBufferMemory
from rapidfuzz import fuzz
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher

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
    def __init__(self):
        try:
            self.intent_labels, self.intent_metadata = self.load_intent_labels_metadata()
            self.intent_encoder = SentenceTransformer("all-mpnet-base-v2")  # Fast, accurate, light
            self.intent_label_embeddings = self.intent_encoder.encode(self.intent_labels, convert_to_tensor=True)

            zs_model = "joeddav/xlm-roberta-large-xnli"
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=AutoModelForSequenceClassification.from_pretrained(zs_model),
                tokenizer=AutoTokenizer.from_pretrained(zs_model, use_fast=False)
            )

            # Primary NER model
            self.ner_pipeline_primary = pipeline(
                "ner", 
                model="dslim/bert-base-NER", 
                tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER", use_fast=True), 
                aggregation_strategy="simple"
            )

            # Secondary fallback NER model (Multilingual(10), fast & compatible) (XLM-R uses SentencePiece → must set use_fast=False)
            fallback_model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-base-ner-hrl")
            fallback_tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-base-ner-hrl", use_fast=False)
            self.ner_pipeline_fallback = pipeline(
                "ner",
                model=fallback_model,
                tokenizer=fallback_tokenizer,
                aggregation_strategy="simple"
            )

            # Tertiary fallback: Indian regional languages (IndicBERT also uses SentencePiece → use_fast=False)
            indic_model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
            indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER", use_fast=False)
            self.ner_pipeline_indic = pipeline(
                "ner",
                model=indic_model,
                tokenizer=indic_tokenizer,
                aggregation_strategy="simple"
            )
            
            logger.info("NLUProcessor initialized successfully with bi-encoder intent classifier.")

        except Exception as e:
            logger.error(f"Failed to initialize NLU models: {e}")
            self.intent_encoder = None
            self.intent_label_embeddings = None
            self.zero_shot_classifier = None
            self.ner_pipeline_primary = None
            self.ner_pipeline_fallback = None
            self.ner_pipeline_indic = None

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


    def classify_intent(self, query):
        """
        Predicts the intent label for a given query using bi-encoder, zero-shot, and rule-based fallback.
        """
        try:
            # Step 0: Hardcoded listing override
            pattern_override = self.detect_list_override(query)
            if pattern_override:
                logger.info("[Intent Override] List pattern matched.")
                return pattern_override

            # Step 1: Bi-Encoder
            if self.intent_encoder and self.intent_label_embeddings is not None:
                intent, score = self._predict_bi_encoder_intent(query)
                if score >= 0.5:
                    return self.override_intent_if_needed(query, intent)
                else:
                    logger.info(f"[Bi-Encoder Score Low] Falling back to Zero-Shot. Score: {score:.2f}")

            # Step 2: Zero-shot fallback
            intent, score = self._predict_zero_shot_intent(query)
            if score >= 0.5:
                return self.override_intent_if_needed(query, intent)
            else:
                logger.info(f"[Zero-Shot Score Low] Falling back to rule-based. Score: {score:.2f}")

            # Step 3: Rule-based fallback
            return detect_task_type_rule_based(query)
        except Exception as e:
            logger.error(f"[Intent Classification Error] {e}")
            return detect_task_type_rule_based(query)

    def _predict_bi_encoder_intent(self, query):
        query_embedding = self.intent_encoder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.intent_label_embeddings)[0]
        best_index = scores.argmax().item()
        best_score = scores[best_index].item()
        best_intent = self.intent_labels[best_index]

        logger.info(f"[Bi-Encoder Intent] '{query}' → {best_intent} (score: {best_score:.2f})")
        return best_intent, best_score

    def _predict_zero_shot_intent(self, query):
        result = self.zero_shot_classifier(query, candidate_labels=self.intent_labels)
        best_intent = result['labels'][0]
        best_score = result['scores'][0]
        logger.info(f"[Zero-Shot Intent] '{query}' → {best_intent} (score: {best_score:.2f})")
        return best_intent, best_score

    def detect_list_override(self, query: str) -> str:
        q = query.lower()
        if re.search(r'\blist\b.*\bdoctor(s)?\b', q):
            return "listing_specific"
        if re.search(r'\blist\b.*\bdepartment(s)?\b', q):
            return "listing_all"
        if re.search(r'\blist\b.*\bservice(s)?\b', q):
            return "listing_specific"
        if re.search(r'\blist\b.*\broom(s)?\b', q):
            return "listing_specific"
        if re.search(r'\b(show|which|available)\b.*\b(service|doctor|department|room)s?\b', q):
            return "listing_specific" or "listing_all"
        return None


# === CONVERSATION MEMORY MODULE ===
class ConversationMemory:
    def __init__(self, max_history_turns=10, summary_threshold=15):
        self.history = []
        self.contextual_entities = []
        self.current_topic = None
        self.last_entity_by_type = {}
        self.max_history_turns = max_history_turns
        self.summary_threshold = summary_threshold

    def add_turn(self, user_query, assistant_response, extracted_entities_map):
        turn_index = len(self.history)
        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "turn_index": turn_index,
            "entities": extracted_entities_map
        })

        if extracted_entities_map:
            for entity_type, entity_list in extracted_entities_map.items():
                for entity_value in entity_list:
                    if entity_value:
                        self.contextual_entities.append({
                            "value": entity_value,
                            "type": entity_type,
                            "turn_index": turn_index
                        })
                        # Track latest entity of each type
                        self.last_entity_by_type[entity_type] = entity_value
            if self.contextual_entities:
                self.current_topic = self.contextual_entities[-1]

        # Prune history if it exceeds limits
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def get_last_entity_by_priority(self, type_priority=None):
        if not type_priority:
            type_priority = ["doctors", "departments", "rooms", "services", "buildings", "floors"]

        # Support single string as input
        if isinstance(type_priority, str):
            type_priority = [type_priority]

        for entity_type in type_priority:
            for turn in reversed(self.history):
                entities = turn.get("entities", {})
                if entities and entity_type in entities and entities[entity_type]:
                    return entities[entity_type][-1]  # Return the last mentioned of that type
        return None


    def get_contextual_history_text(self, num_turns=5):
        history_text = ""
        recent_turns = self.history[-num_turns:]

        for turn in recent_turns:
            user_msg = turn.get("user", "[no user input]")
            assistant_msg = turn.get("assistant", "[no assistant response]")
            turn_index = turn.get("turn_index", -1)
            history_text += f"User (Turn {turn_index}): {user_msg}\nAssistant (Turn {turn_index}): {assistant_msg}\n"

        return history_text.strip()

    def get_relevant_entities_from_recent_turns(self, turns_to_check=3):
        relevant = []
        if not self.history or not self.contextual_entities:
            return []
        
        current_turn_index = self.history[-1]["turn_index"]
        
        for entity_info in reversed(self.contextual_entities):
            if (current_turn_index - entity_info["turn_index"]) < turns_to_check:
                if not any(r['value'] == entity_info['value'] and r['type'] == entity_info['type'] for r in relevant):
                    relevant.append(entity_info)
            else:
                break
        return list(reversed(relevant))
    def get_recent_entities_by_type(self, entity_type, max_turns=5):
        current_index = self.history[-1]["turn_index"] if self.history else 0
        recent = [
            ent["value"]
            for ent in reversed(self.contextual_entities)
            if ent["type"] == entity_type and (current_index - ent["turn_index"] <= max_turns)
        ]
        return list(dict.fromkeys(recent))  # remove duplicates, preserve order
    

class RedisUserMemoryStore:
    def __init__(self, redis_url="redis://localhost:6379", use_summary=False):
        self.redis_url = redis_url
        self.use_summary = use_summary
        self.lock = threading.Lock()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")  # Summarization LLM

    def get(self, user_id):
        with self.lock:
            chat_history = RedisChatMessageHistory(url=self.redis_url, session_id=user_id)

            if self.use_summary:
                from langchain.memory import ConversationSummaryBufferMemory
                return ConversationSummaryBufferMemory(
                    llm=self.llm,
                    chat_memory=chat_history,
                    memory_key="chat_history",
                    return_messages=True,
                    max_token_limit=1024  # we can tune this
                )
            else:
                return ConversationBufferMemory(
                    chat_memory=chat_history,
                    memory_key="chat_history",
                    return_messages=True
                )

    def clear(self, user_id):
        with self.lock:
            chat_history = RedisChatMessageHistory(url=self.redis_url, session_id=user_id)
            chat_history.clear()

    def all_user_ids(self):
        # Redis does not expose all keys safely in production unless scanned.
        return ["redis_does_not_track_keys_globally"]

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


# === ENHANCED HOSPITAL DATA LOADER WITH QA PAIRS ===
class HospitalDataLoader:
    def __init__(self, hospital_filepath=HOSPITAL_MODEL_JSON_PATH, qa_filepath=QA_PAIRS_JSON_PATH):
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


# Enhanced NLU Processor to handle Hospital and QA content
class EnhancedNLUProcessor(NLUProcessor):
    def override_intent_if_needed(self, query, base_intent):
        """
        Apply hospital-specific override rules for better precision.
        """
        query_l = query.lower()

        if any(term in query_l for term in ['where', 'locate', 'located at', 'located in']) and any(term in query_l for term in ['room', 'ward', 'floor', 'building']):
            return "location"
        elif any(term in query_l for term in ['where', 'find', 'locate']) and any(term in query_l for term in ['department', 'unit']):
            return "location"
        elif any(term in query_l for term in ['where', 'find', 'who is', 'meet']) and any(term in query_l for term in ['doctor', 'dr', 'physician', 'specialist']):
            return "location"
        elif any(term in query_l for term in ['when', 'availability']) and any(term in query_l for term in ['doctor', 'dr', 'consultant']):
            return "doctor_availability"
        elif any(term in query_l for term in ['contact', 'phone', 'email', 'call']):
            return "contact_info"
        elif any(term in query_l for term in ['appointment', 'book', 'schedule', 'register']):
            return "booking_info"
        elif any(term in query_l for term in ['time', 'timing', 'open', 'close', 'hours', 'schedule']) and "doctor" not in query_l:
            return "operating_hours"
        elif any(term in query_l for term in ['service', 'facility', 'treatment', 'procedure']):
            return "service_info"

        return base_intent


data_loader = HospitalDataLoader()

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

def clean_extracted_entities(entities):
    """
    Cleans broken or subword tokens from entity lists like doctors/persons/etc.
    - Removes '##' prefixes (BERT-style subwords)
    - Joins fragments into full names (e.g., 'Shr', '##uti', 'Sharma' → 'Shruti Sharma')
    - Handles prefixes like 'Dr.' or 'Prof.'
    - Deduplicates and trims
    """
    name_prefixes = {"dr", "dr.", "prof", "prof.", "mr", "ms", "mrs"}

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
    entity_map = {"doctors": [], "organizations": [], "locations": [], "dates": [], "times": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entity_map["doctors"].append(ent.text)
        elif ent.label_ in {"ORG"}:
            entity_map["organizations"].append(ent.text)
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

# Fast Reranker (first stage)
def rerank_documents_fast(query, docs, top_k=8):
    if not fast_reranker:
        logger.warning("Fast reranker not available. Returning top documents without reranking.")
        return docs[:top_k]

    if not docs or len(docs) < 2:
        logger.info("Not enough documents for reranking. Returning as-is.")
        return docs[:top_k]

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = fast_reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.info(f"fast reranker reranked {len(docs)} docs, returning top {top_k}.")
        return [doc for score, doc in scored_docs[:top_k]]
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

SYNONYM_MAP = {
    # Facilities & Navigation
    "elevator": ["lift"], "lift": ["elevator"],
    "toilet": ["washroom", "restroom", "lavatory", "wc"],
    "washroom": ["toilet"], "restroom": ["toilet"], "lavatory": ["toilet"], "wc": ["toilet"],
    "stairs": ["staircase"], "staircase": ["stairs"],
    "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0", "gf"], "gf": ["ground floor"],
    "building": ["hospital building"], "hospital building": ["building"],
    "waiting room": ["waiting area"], "waiting area": ["waiting room"],
    "ward": ["room"], "room": ["ward"],

    # Contact & Timings
    "contactno": ["contact number", "phone no", "phone number", "contact"],
    "timings": ["operating hours", "open hours", "availability", "schedule"],
    "operating hours": ["timings"], "open hours": ["timings"], "schedule": ["timings"],

    # People
    "doctor": ["dr", "dr.", "physician", "consultant", "specialist", "professor", "doc"],
    "dr.": ["doctor"], "physician": ["doctor"], "doc": ["doctor"],
    "staff": ["hospital staff"], "hospital staff": ["staff"],

    # Departments / Specialties
    "opd": ["outpatient department", "out-patient department", "clinic"],
    "er": ["emergency room", "emergency", "casualty"],
    "emergency room": ["er"], "casualty": ["emergency"],

    "icu": ["intensive care unit"], "intensive care unit": ["icu"],

    "cardiology": ["heart department", "cardiac"], "heart": ["cardiology"], "cardiac": ["cardiology"],
    "neurology": ["neuro department", "nerve specialist"], "brain": ["neurology"],
    "pediatrics": ["child care", "paediatrics", "peds"], "children": ["pediatrics"],
    "orthopedics": ["bone department", "ortho"], "bones": ["orthopedics"],
    "radiology": ["x-ray department", "imaging"], "scan": ["radiology"],
    "dermatology": ["skin"], "skin": ["dermatology"],
    "gynecology": ["women", "obgyn"], "women": ["gynecology"],
    "ophthalmology": ["eye"], "eye": ["ophthalmology"],
    "psychiatry": ["psychiatrist"], "psychiatrist": ["psychiatry"],
    "oncology": ["cancer"], "cancer": ["oncology"],
    "anesthesia": ["anesthesiology"], "anesthesiology": ["anesthesia"],

    # Services
    "checkup": ["diagnosis"], "test": ["diagnosis"],
    "lab": ["laboratory"], "laboratory": ["lab"],
    "mri scan": ["mri"], "ct scan": ["ct"], "ultrasound scan": ["ultrasound"],
    "report": ["diagnosis report"],

    # Booking
    "appointment": ["booking", "reservation", "appt"],
    "booking": ["appointment"],

    # Equipment
    "ventilator machine": ["ventilator"], "ventilator": ["ventilator machine"],
    "wheel chair": ["wheelchair"], "wheelchair": ["wheel chair"],

    # Institution
    "aiims": ["all india institute of medical sciences"],
    "all india institute of medical sciences": ["aiims"]
}


def expand_query_with_synonyms(query): # Logic remains the same, uses updated SYNONYM_MAP
    query_lower = query.lower()
    variants = set([query_lower])

    # Apply global substitution for any matching key or synonym
    for keyword, synonyms_list in SYNONYM_MAP.items():
        all_variants_for_keyword = [keyword] + synonyms_list 
        for current_variant_text in list(variants): # Iterate over a copy as variants set might change
            for term_to_find in all_variants_for_keyword:
                pattern = re.compile(rf'\b{re.escape(term_to_find)}\b', re.IGNORECASE)
                if pattern.search(current_variant_text):
                    for replacement_term in all_variants_for_keyword:
                        if replacement_term.lower() != term_to_find.lower(): 
                            new_query_variant = pattern.sub(replacement_term, current_variant_text)
                            variants.add(new_query_variant)
    final_variants = list(set(v for v in variants if len(v) > 3))[:6]
    if len(final_variants) > 1 and query_lower not in final_variants : # Log only if actual expansion happened
         logger.info(f"Expanded query '{query}' to variants: {final_variants}")
    elif not final_variants : # if query was too short or no expansions
        final_variants = [query_lower]

    return final_variants


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

def detect_and_translate(text, target_lang="en"):
    try:
        # Heuristic: If text contains only basic ASCII letters and spaces, skip translation
        if re.fullmatch(r"[a-zA-Z0-9\s\?\.,\-']+", text.strip()):
            return text, "en"
        
        detected_lang = detect(text)
        if detected_lang == target_lang: 
            return text, detected_lang
        
        translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
        logger.info(f"Translated query from {detected_lang} to {target_lang}: '{text}' -> '{translated_text}'")
        return translated_text, detected_lang
    except Exception as e:
        logger.warning(f"Language detection/translation failed for '{text}': {e}. Using original text.")
        return text, "en"

def detect_target_language_for_response(query):
    language_map = {"hindi": "hi", "punjabi": "pa", "tamil": "ta", "telugu": "te", "kannada": "kn", "marathi": "mr", "bengali": "bn", "urdu": "ur", "gujarati": "gu", "malayalam": "ml", "english": "en", "spanish": "es", "french": "fr", "german": "de", "russian": "ru"}
    query_lower = query.lower()
    for lang_name, lang_code in language_map.items():
        if re.search(rf'\bin\s+{re.escape(lang_name)}\b', query_lower):
            cleaned_query = re.sub(rf'\s*\bin\s+{re.escape(lang_name)}\b', '', query_lower, flags=re.IGNORECASE).strip()
            logger.info(f"Detected target response language: {lang_name} ({lang_code}). Cleaned query: '{cleaned_query}'")
            return cleaned_query, lang_code
    return query, None

def extract_entities_rule_based(query):
    query_lower = normalize_query(query)
    entities = {
        "hospitals": [], "buildings": [], "floors": [], "rooms": [],
        "departments": [], "doctors": [], "services": [],
        "lifts": [], "stairs": [], "washrooms": [], "general_terms": []
    }

    # Hospital Names (example, make more robust or use known list)
    hospital_matches = re.findall(r'\b(aiims(?:\s+\w+)?|apollo|fortis|max\s+healthcare|city\s+hospital)\b', query_lower, re.IGNORECASE)
    for m in hospital_matches: 
        entities["hospitals"].append(m.strip())
    if not entities["hospitals"] and "hospital" in query_lower : 
        entities["hospitals"].append("hospital")


    # Building/Block/Wing
    building_matches = re.findall(r'\b(block\s*[\w\d-]+|building\s*[\w\d-]*|tower\s*[\w\d-]*|wing\s*[\w\d-]+|diagnostic\s*block)\b', query_lower, re.IGNORECASE)
    for m in building_matches: 
        entities["buildings"].append(m.strip())

    # Floor Numbers
    floor_matches = re.findall(r'(?:floor|level|flr)\s*(\d+[-\w]*\b)|(\b\d+)(?:st|nd|rd|th)?\s*(?:floor|level|flr)|(ground\s*floor|gf\b)', query_lower, re.IGNORECASE)
    for m_tuple in floor_matches:
        val = next(filter(None, m_tuple), None) # Get the first non-None value from the tuple
        if val:
            if "ground floor" in val or "gf" == val : 
                entities["floors"].append("0")
            else: 
                entities["floors"].append(re.sub(r'[^\d\w-]', '', val)) # Clean non-alphanumeric except dash

    # Room Numbers/Names (generic, may need context)
    room_matches = re.findall(r'\b(?:room|rm|cabin|opd)\s*([\w\d-]+)\b|(\b\d+[A-Za-z]?-?\d*[A-Za-z]?\b(?!\s*floor|\s*st|\s*nd|\s*rd|\s*th|\s*am|\s*pm))', query_lower, re.IGNORECASE)
    for m_tuple in room_matches:
        val = next(filter(None, m_tuple), None)
        if val and len(val) > 1 : 
            entities["rooms"].append(val.strip())

    # Other known room/function/facility types (custom list)
    room_function_keywords = [
        "registration", "reception", "waiting area", "consultation room",
        "nursing station", "admin office", "medical superintendent", "dean office",
        "central records", "laboratory", "lab", "x-ray room", "ultrasound room",
        "radiology room", "pathology lab", "pharmacy", "cafeteria", "canteen",
        "icu", "ward", "emergency", "diagnostic center", "treatment room", 
        "vaccination center", "procedure room", "assessment room", "staff room"
    ]

    for keyword in room_function_keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
            entities["rooms"].append(keyword.title())
            # Also treat as department if it's a dual-role entity
            if keyword in ["opd", "icu", "emergency", "pharmacy", "pathology", "radiology"]:
                entities["departments"].append(keyword.lower())

    # Departments (extend with more common department names)
    department_keywords = [
        "cardiology", "neurology", "oncology", "pediatrics", "paediatrics", "radiology", "surgery", "opd", "outpatient",
        "emergency", "casualty", "icu", "intensive care", "orthopedics", "ortho", "gynecology", "gynaecology",
        "dermatology", "ent", "ear nose throat", "urology", "psychiatry", "pathology", "laboratory", "lab",
        "pharmacy", "physiotherapy", "anesthesia", "dental", "opthalmology", "eyes"
    ]
    for dept_kw in department_keywords:
        if re.search(rf'\b{dept_kw}\b', query_lower, re.IGNORECASE):
            entities["departments"].append(dept_kw)
    # Also match "cardiology department", "opd clinic" etc.
    dept_matches = re.findall(r'\b(' + '|'.join(department_keywords) + r')\s*(?:department|dept|clinic|unit|ward|center|section)\b', query_lower, re.IGNORECASE)
    for m in dept_matches: 
        entities["departments"].append(m.strip())

    # 1. Detect names prefixed with "Dr." or "Doctor"
    prefix_matches = re.findall(r'\b(?:dr\.?|doctor)\s+([a-z][a-z\s\.-]{2,})\b', query_lower, re.IGNORECASE)
    for name in prefix_matches:
        cleaned = re.sub(r'\b(department|clinic|hospital|ward|unit)\b', '', name, flags=re.IGNORECASE).strip()
        if len(cleaned.split()) >= 1 and len(cleaned) > 3:
            entities["doctors"].append(cleaned.title())

    # 2. Match known doctor names from data using fuzzy match
    if data_loader and data_loader.all_known_entities:
        known_doctors = data_loader.all_known_entities.get("doctors", [])
        for doc_name in known_doctors:
            doc_name_clean = re.sub(r'\bdr\.?\s*', '', doc_name, flags=re.IGNORECASE).strip()

            if all(part in query_lower for part in doc_name_clean.lower().split()):
                entities["doctors"].append(doc_name)
                continue

            score_full = fuzz.token_sort_ratio(query_lower, doc_name_clean.lower())
            score_partial = fuzz.partial_ratio(query_lower, doc_name_clean.lower())
            score = max(score_full, score_partial)

            score = fuzz.token_sort_ratio(query_lower, doc_name_clean.lower())
            if score >= 80:
                entities["doctors"].append(doc_name)


    # 3. Extract possible name-like entities heuristically (if not found above)
    if not entities["doctors"]:
        name_like_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', query, re.UNICODE)
        for name in name_like_matches:
            if len(name.split()) >= 2 and len(name) > 5:
                entities["doctors"].append(name.strip())

    # Services (example keywords, extend significantly)
    service_keywords = ["x-ray", "mri", "ct scan", "ultrasound", "ecg", "blood test", "consultation", "therapy", "checkup", "vaccination", "dialysis", "angiography", "biopsy", "endoscopy", "echo", "tmt", "blood sugar test", "cbc", "serology"]
    for svc_kw in service_keywords:
        if re.search(rf'\b{re.escape(svc_kw)}\b', query_lower, re.IGNORECASE): # escape for "ct scan"
            entities["services"].append(svc_kw)

    # Lifts, Stairs, Washrooms (generic), doctor
    if re.search(r'\b(lift|elevator)\b', query_lower): 
        entities["lifts"].append("lift")
    if re.search(r'\b(stairs|staircase)\b', query_lower): 
        entities["stairs"].append("stairs")
    if re.search(r'\b(washroom|toilet|restroom|lavatory|wc)\b', query_lower): 
        entities["washrooms"].append("washroom")
    if re.search(r"\b(?:\d+)?\s*(?:doctors|doctor|physicians|consultants|specialists)\b", query_lower):
        entities["doctors"].append("ANY_DOCTOR")  # or just "doctor"

    # Boost from known entities loaded from data
    if data_loader and data_loader.all_known_entities:
        for entity_type, known_list in data_loader.all_known_entities.items():
            if entity_type in entities: # Only for types we are extracting
                for known_item in known_list:
                    # Exact match (case insensitive)
                    if re.search(rf'\b{re.escape(known_item)}\b', query_lower, re.IGNORECASE):
                        entities[entity_type].append(known_item)
                    # Check synonyms for this known_item
                    for syn_keyword, syn_list in SYNONYM_MAP.items():
                        if known_item.lower() == syn_keyword.lower() or known_item.lower() in [s.lower() for s in syn_list]:
                            for form in [syn_keyword] + syn_list:
                                if re.search(rf'\b{re.escape(form)}\b', query_lower, re.IGNORECASE):
                                    entities[entity_type].append(known_item)
                                    break
                            break


    for k in entities: 
        entities[k] = sorted(list(set(e.strip() for e in entities[k] if e and len(e.strip()) > 1))) # Deduplicate and clean
    
    # Remove "dr" or "doctor" if it's the only thing in doctors list and other doctor names are present
    if "dr" in entities["doctors"] and len(entities["doctors"]) > 1: 
        entities["doctors"].remove("dr")
    if "doctor" in entities["doctors"] and len(entities["doctors"]) > 1: 
        entities["doctors"].remove("doctor")
    logger.info(f"[RuleBased NER] Final extracted entities for '{query}': {entities}")
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
    Returns: best_score, best_match_doc, matched_field
    """

    best_score = 0
    best_doc = None
    matched_field = None

    for doc in docs:
        fields_to_check = list(doc.metadata.keys()) + ["page_content"]
        for field in fields_to_check:
            val = doc.metadata.get(field) if field != "page_content" else doc.page_content
            if not isinstance(val, str): 
                continue
            try:
                score = semantic_similarity(entity_value, val)
                if score > best_score:
                    best_score = score
                    best_doc = doc
                    matched_field = field
            except Exception as e:
                logger.warning(f"Semantic similarity error for field '{field}': {e}")
                continue

    return best_score, best_doc, matched_field

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


def extract_doctor_name(text: str) -> str:
    """
    Extracts a doctor's name like 'Dr Gaurav Gupta' or 'Dr. Shruti Sharma' from a query.
    """
    try:
        match = re.search(r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text, flags=re.IGNORECASE)
        return match.group(0).strip() if match else ""
    except Exception as e:
        print(f"[extract_doctor_name error]: {e}")
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
         "how are you", "how’s it going", "what’s up", "wassup", 
         "bored", "i’m back", "doing nothing", "tell me something", 
         "interesting", "just checking", "just saying hi",
         "hi again", "you awake?", "you online?", "mood off", "i’m tired", "i’m bored", 
         "anything new?", "say something", "tell me a joke", "reply pls", "pls respond",
         "ok", "okay", "cool", "sure", "fine", "great", "nice", "good", "awesome", "super" # Added common affirmations often part of small talk
    ]
    appreciation_variants = [
        "thank you", "thanks", "thx", "ty", "tysm", # Moved thanks here
        "you are doing good", "good job", "great work", "well done", "very well", 
        "appreciate it", "thanks a lot", "thank you so much", "that’s helpful", 
        "amazing answer", "awesome reply", "you nailed it", "you’re awesome", 
        "you rock", "brilliant", "excellent", "superb", "love that", "fantastic", 
        "mind blowing", "next level", "exactly what I needed", "so quick", "so smart"
    ]
    confirmation_variants = [
        "yes", "yep", "yeah", "sure", "absolutely", "of course", "definitely", 
        "yup", "you got it", "correct", "right", "exactly", "that’s right", "alright", "indeed"
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
    q = query.lower().strip()
    q = collapse_repeated_letters(q)

    # Standard replacements
    q = q.replace("dept.", "department")
    q = q.replace("dr.", "doctor")

    # Room/OPD normalization (e.g., "opd3", "room 101a")
    q = re.sub(r"\b(opd|room|rm|cabin|ward|icu)\s*(\d+[a-z]?)", r"\1-\2", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(\d+[a-z]?)\s*(opd|room|rm|cabin|ward|icu)", r"\2-\1", q, flags=re.IGNORECASE)

    # Specific case: lift lobby
    q = re.sub(r"lift\s*lobby[-\s]*(\d+)", r"lift lobby \1", q)

    # Collapse repeated characters again just in case
    q = re.sub(r"(.)\1{2,}", r"\1", q)

    # Normalize room codes
    tokens = q.split()
    normalized_tokens = []
    for token in tokens:
        if is_likely_room_code(token):
            normalized_tokens.append(normalize_room_code(token))
        else:
            normalized_tokens.append(token)
    q = " ".join(normalized_tokens)

    # Punctuation cleanup (allow only '.', '-', '@')
    q = re.sub(r"[^\w\s\-\.@]", "", q)

    # Whitespace cleanup
    q = re.sub(r"\s+", " ", q).strip()

    # Normalize plural terms to singular for entity detection
    q = re.sub(r"\bdoctors\b", "doctor", q)
    q = re.sub(r"\bspecialists\b", "specialist", q)
    q = re.sub(r"\bconsultants\b", "consultant", q)
    q = re.sub(r"\bdepartments\b", "department", q)
    q = re.sub(r"\bservices\b", "service", q)

    # Standardize medical abbreviations
    abbrev_map = {
        "opd": "outpatient department",
        "icu": "intensive care unit",
        "ot": "operation theatre",
        "ent": "ear nose throat",
        "xray": "x-ray"
    }
    for abbr, full in abbrev_map.items():
        q = re.sub(rf"\b{abbr}\b", full, q)

    # Normalize number words
    num_map = {"two": "2", "three": "3", "four": "4", "five": "5"}
    for word, digit in num_map.items():
        q = re.sub(rf"\b{word}\b", digit, q)

    return q

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
    original_query = query.strip()
    rewritten_query = original_query
    query_lower_normalized = normalize_query(original_query.lower()) # Use normalized for matching

    context_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
    
    # Priority: doctor > department > service > room > hospital/building
    salient_topic_entity_value = memory.get_last_entity_by_priority(
        type_priority=["doctors", "departments", "services", "rooms", "hospitals", "buildings"]
    )
    salient_topic_type = None
    if salient_topic_entity_value:
        for entity_type in ["doctors", "departments", "services", "rooms", "hospitals", "buildings"]:
            if memory.last_entity_by_type.get(entity_type) == salient_topic_entity_value:
                salient_topic_type = entity_type
                break
    
    # Fallbacks if no priority entity found
    if not salient_topic_entity_value and memory.current_topic:
        salient_topic_entity_value = memory.current_topic.get("value")
        salient_topic_type = memory.current_topic.get("type")
    elif not salient_topic_entity_value and context_entities:
        salient_topic_entity_value = context_entities[-1].get("value")
        salient_topic_type = context_entities[-1].get("type")

    follow_up_keywords = [
        "contact", "email", "phone", "website", "location", "address", "services", "specialty",
        "availability", "schedule", "timings", "hours", "visiting hours", "profile", "about",
        "department", "room", "floor"
    ]
    follow_up_pattern_str = r"^(and|also|what about|how about|tell me more about|more info on)\b.*(" + "|".join(follow_up_keywords) + \
                           r")?|^(their|his|her|its)\b.*(" + "|".join(follow_up_keywords) + \
                           r")|^\b(" + "|".join(follow_up_keywords) + r")\b"
    
    is_short_follow_up_keyword_only = len(original_query.split()) <= 2 and any(kw in query_lower_normalized for kw in follow_up_keywords)


    if salient_topic_entity_value and (re.search(follow_up_pattern_str, query_lower_normalized, re.IGNORECASE) or is_short_follow_up_keyword_only) :
        if not re.search(rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', query_lower_normalized, re.IGNORECASE): # Check first word of entity
            rewritten_query = f"{salient_topic_entity_value} - {original_query}"
            logger.info(f"[Coref Rewrite - Follow-up] Rewrote '{original_query}' → '{rewritten_query}' using salient entity '{salient_topic_entity_value}'")
            return rewritten_query

    if len(original_query.split()) < 5 and salient_topic_entity_value:
        # Avoid rewriting if salient entity type is too generic or query is a question word
        if salient_topic_type in ["floors", "buildings"] or original_query.lower().startswith(("what","who","where","when","why","how")):
             if rewritten_query != original_query: logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
             return rewritten_query
        
        rewritten_query = resolve_pronouns_with_spacy(original_query, salient_topic_entity_value)


    if rewritten_query != original_query:
        logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
        return rewritten_query
    # LLM fallback if enabled and rewrite didn’t happen
    if rewritten_query == original_query and salient_topic_entity_value and USE_LLM_INTENT_FALLBACK:
        try:
            prompt = f"""
You are a helpful assistant rewriting user queries using recent context.

The user previously mentioned: "{salient_topic_entity_value}"

Please rewrite the following query by resolving any references (like 'it', 'they', 'their', etc.) to this context if appropriate.
If no rewrite is needed, just return the original query.

Query: "{original_query}"
Rewritten:
"""
            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.1)
            llm_response = llm.invoke(prompt).content.strip()
            if llm_response and llm_response.lower() != original_query.lower():
                logger.info(f"[LLM Coref Rewrite] '{original_query}' → '{llm_response}'")
                return llm_response
        except Exception as e:
            logger.warning(f"[LLM Coref Rewrite Fallback Failed] {e}")

    return rewritten_query


def handle_small_talk(user_query: str, memory, user_id: str, detected_input_lang: str = "en", convo_intent: str = None)-> dict:
    import random
    query_corrected = correct_spelling(user_query)
    query_cleaned = collapse_repeated_letters(query_corrected)

    has_prior_convo = memory and len(memory.history) > 1
    personalization_prefix = "Welcome back! 😊" if convo_intent == "greeting" and has_prior_convo else ""


    convo_prompt = f"""
You are a warm, friendly assistant working at AIIMS Jammu. 
The user is speaking casually — respond with kindness, sometimes humor, and use emojis naturally.
Avoid repeating yourself. Always vary your response. 
Add a touch of human-like warmth and keep it short and refreshing.

{personalization_prefix}
User: {query_cleaned}
Assistant:"""

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="gemma-2-9b-it", temperature=0.85)
        response = llm.invoke(convo_prompt).content.strip()
        # Applied max length guard
        if len(response) > 500:
            response = response[:500] + "..."

        # Emoji variety
        emoji_pool = ["😊", "👋", "😄", "🌟", "🙂", "🙌", "👍", "🤗"]
        if not any(emoji in response for emoji in emoji_pool):
            response += " " + random.choice(emoji_pool)
    except Exception as e:
        logger.error(f"Error in small talk handler: {e}")
        response = random.choice([
            "Hey visitor! 👋 What can I help you with at Aimms Jammu?",
            "Hi! 😊 Feel free to ask about departments or doctors.",
            "I'm here! What info are you looking for today?"
        ])

    # Check if response is identical to previous one to avoid repetition
    if memory and memory.history:
        last_reply = memory.history[-1]["assistant"]
        if response.strip() == last_reply.strip():
            response += " 😊 Anything else I can assist you with?"

    # Save small talk to memory
    memory.add_turn(query_cleaned, response, extracted_entities_map={})
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
            "intent": convo_intent
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
        "emergency medicine", "nephrology", "urology", "gastroenterology",

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
    extracted_query_entities = nlu_processor.extract_entities(processed_query)
    extracted_query_entities = clean_extracted_entities(extracted_query_entities)
    # Enhance with spaCy NER HERE
    spacy_entities = extract_spacy_ner_entities(processed_query)
    extracted_query_entities = merge_entities(extracted_query_entities, spacy_entities)
    # Enhance with PhraseMatcher-based entities from spaCy
    spacy_matched_entities = extract_entities_spacy(processed_query)
    extracted_query_entities = merge_entities(extracted_query_entities, spacy_matched_entities)

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
    task_type = nlu_processor.classify_intent(processed_query)
    convo_intent = detect_conversational_intent(processed_query)

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
        if target_lang_code and target_lang_code != "en": 
            out_of_scope_response = GoogleTranslator(source="en", target=target_lang_code).translate(out_of_scope_response)
        elif detected_input_lang != "en": 
            out_of_scope_response = GoogleTranslator(source="en", target=detected_input_lang).translate(out_of_scope_response)
        return {"answer": out_of_scope_response, "debug_info": {"task_type": task_type}}

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
        doc_id = doc.metadata.get("source_doc_id", doc.page_content[:30])
        unique_docs[doc_id] = doc
    retrieved_docs = list(unique_docs.values())

    logger.info(f"[Synonym Retrieval] Total unique docs after expansion: {len(retrieved_docs)}")
    if not retrieved_docs:
        logger.warning(f"No documents retrieved for hospital query: {processed_query}")
        clarification_msg = "I couldn't find specific information for your query. "
        suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
        clarification_msg += " ".join(suggestions) if suggestions else "Could you try rephrasing or provide more details?"
        conv_memory.history[-1]["assistant"] = clarification_msg
        user_memory_store.save(user_id, conv_memory)
        if target_lang_code and target_lang_code != "en": 
            clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en": 
            clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)
        return {"answer": clarification_msg, "related_queries": suggestions if suggestions else []}

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
    if missing_entities and task_type != "general_information":
        logger.warning(f"[Missing Context] Could not find these terms in retrieved docs: {missing_entities}")
        missing_list = ", ".join(missing_entities)
        clarification_msg = f"I couldn't find specific information about: {missing_list}. Could you clarify or rephrase your query?"

        # Save to memory
        conv_memory.history[-1]["assistant"] = clarification_msg
        user_memory_store.save(user_id, conv_memory)

        # Translate if needed
        if target_lang_code and target_lang_code != "en":
            clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en":
            clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)

        return {"answer": clarification_msg}
            

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
   - If the answer refers to a specific person or entity mentioned earlier, restate the name for clarity (e.g., “Dr. Aymen Masood is located in…”).

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