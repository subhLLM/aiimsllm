from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from transformers import pipeline, AutoTokenizer
import torch
from collections import Counter
import sys
import io
import threading

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

# === NLU PROCESSOR ===
class NLUProcessor:
    def __init__(self):
        try:
            # Bi-Encoder for intent classification
            self.intent_labels = [
                "location", "contact_info", "booking_info", "explanation", "comparison",
                "operating_hours", "listing_all", "listing_specific", "out_of_scope", "general_information",
                "doctor_availability", "department_info", "service_info"
            ]
            self.intent_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, accurate, light
            self.intent_label_embeddings = self.intent_encoder.encode(self.intent_labels, convert_to_tensor=True)

            # Primary NER model
            self.ner_pipeline_primary = pipeline(
                "ner", 
                model="dslim/bert-base-NER", 
                tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER", use_fast=True),
                aggregation_strategy="simple"
            )
            
            # Secondary fallback NER model (Multilingual, fast & compatible)
            self.ner_pipeline_fallback = pipeline(
                "ner", 
                model="Davlan/bert-base-multilingual-cased-ner-hrl", 
                tokenizer=AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl", use_fast=True),
                aggregation_strategy="simple"
            )
            
            logger.info("NLUProcessor initialized successfully with bi-encoder intent classifier.")

        except Exception as e:
            logger.error(f"Failed to initialize NLU models: {e}")
            self.intent_encoder = None
            self.intent_label_embeddings = None
            self.ner_pipeline_primary = None
            self.ner_pipeline_fallback = None

    def classify_intent(self, query):
        if not self.intent_encoder or self.intent_label_embeddings is None:
            logger.warning("Bi-encoder not available. Falling back to rule-based intent detection.")
            return detect_task_type_rule_based(query)

        try:
            query_embedding = self.intent_encoder.encode(query, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, self.intent_label_embeddings)[0]
            best_index = scores.argmax().item()
            best_score = scores[best_index].item()
            best_intent = self.intent_labels[best_index]

            logger.info(f"[Intent] Query: '{query}' → Intent: '{best_intent}' (score: {best_score:.2f})")

            # Optional threshold for out-of-scope fallback
            if best_score < 0.4:
                logger.info("Low intent confidence. Returning 'general_information'.")
                return "general_information"

            return best_intent
        except Exception as e:
            logger.error(f"Error in bi-encoder intent classification: {e}")
            return detect_task_type_rule_based(query)

    def extract_entities(self, query):
        def process_ner_results(ner_results):
            entities = {
                "hospitals": [], "buildings": [], "floors": [], "rooms": [],
                "departments": [], "doctors": [], "services": [],
                "lifts": [], "stairs": [], "washrooms": [], # Washrooms still relevant
                "general_terms": [], "misc": [] # Added misc for PER, ORG not mapped
            }
            for entity in ner_results:
                entity_type = entity.get("entity_group", "").upper() # NER models use upper (e.g., LOC, ORG, PER)
                value = entity.get("word", "").strip()

                if entity_type == "LOC":
                    # Try to be more specific based on keywords
                    if "hospital" in value.lower() or "aiims" in value.lower() : # Assuming AIIMS is a hospital name pattern
                        entities["hospitals"].append(value)
                    elif "building" in value.lower() or "diagnostic" in value.lower() or "wing" in value.lower() or "block" in value.lower() or "tower" in value.lower(): # diagnostic, wing are from example
                        entities["buildings"].append(value)
                    elif "floor" in value.lower() or re.match(r'\d+', value.split()[0] if value else ""): # Check if first part is a number for floor
                        entities["floors"].append(value)
                    elif "room" in value.lower() or re.match(r'\w*\d+\w*', value): # Simple room pattern
                        entities["rooms"].append(value)
                    elif "lift" in value.lower():
                        entities["lifts"].append(value)
                    elif "stair" in value.lower():
                        entities["stairs"].append(value)
                    elif "washroom" in value.lower() or "restroom" in value.lower() or "toilet" in value.lower() or "bathroom" in value.lower():
                        entities["washrooms"].append(value)
                    else:
                        entities["general_terms"].append(value) # Default for LOC if not specific
                elif entity_type == "ORG": # Could be hospital or department
                    if "hospital" in value.lower() or "aiims" in value.lower():
                         entities["hospitals"].append(value)
                    elif any(dept_kw in value.lower() for dept_kw in ["department", "cardiology", "Anesthesiology" "opd", "clinic", "radiology", "surgery"]): # Add more KWs
                        entities["departments"].append(value)
                    else:
                        entities["hospitals"].append(value) # Default ORG to hospital if not clearly department
                elif entity_type == "PER":
                    if value.lower().startswith("dr") or "doctor" in value.lower():
                         entities["doctors"].append(value)
                    else: # General person, could be a doctor name without "Dr."
                         entities["doctors"].append(value) # Tentatively add to doctors
                elif entity_type == "MISC": # General terms or services
                    # Could check against known services if list is available
                    entities["general_terms"].append(value)
                else: # Unmapped entity types
                    entities["misc"].append(value)


            for k in entities:
                entities[k] = sorted(list(set(entities[k]))) # Use list(set()) for deduplication
            return entities

        # Try primary NER
        try:
            if self.ner_pipeline_primary:
                primary_results = self.ner_pipeline_primary(query)
                entities = process_ner_results(primary_results)
                logger.info(f"[NER Primary] Extracted entities: {entities}")


                # If entities are sparse (especially for doctors/departments), try fallback or augment
                if (not entities.get("doctors") and not entities.get("departments")) and self.ner_pipeline_fallback:
                    logger.info(f"[NER Primary] Sparse entities, trying fallback NER.")
                    fallback_results = self.ner_pipeline_fallback(query)
                    fallback_entities = process_ner_results(fallback_results)
                    logger.info(f"[NER Fallback] Extracted entities: {fallback_entities}")
                    # Merge results, giving preference to fallback if it found something primary didn't
                    for key, value_list in fallback_entities.items():
                        if value_list and not entities.get(key):
                            entities[key] = value_list
                        elif value_list:
                            entities[key] = sorted(list(set(entities[key] + value_list)))
                
                # If still sparse, try rule-based as a final augmentation before returning
                if not any(entities.get(k) for k in ["doctors", "departments", "rooms", "hospitals", "washrooms", "offices", "wards", "opd", "clinics"]):
                    logger.info(f"[NER] Still sparse, augmenting with rule-based NER.")
                    rule_based_entities = extract_entities_rule_based(query)
                    for key, value_list in rule_based_entities.items():
                        if value_list and not entities.get(key):
                            entities[key] = value_list
                        elif value_list:
                             entities[key] = sorted(list(set(entities[key] + value_list)))\
                             
                # Heuristically extract and match full doctor names from known list
                if data_loader and data_loader.all_known_entities:
                    known_doctors = data_loader.all_known_entities.get("doctors", [])
                    query_clean = re.sub(r'[^a-zA-Z\s]', '', query).strip().lower()
                    for doc_name in known_doctors:
                        doc_name_clean = doc_name.lower()
                        if query_clean in doc_name_clean or doc_name_clean in query_clean:
                            if doc_name not in entities["doctors"]:
                                entities["doctors"].append(doc_name)

                    # Optional: Clean up malformed tokens like '##hr'
                    entities["doctors"] = [d for d in entities["doctors"] if len(d) > 4 and not d.startswith("##")]

                logger.info(f"[NER Final] Extracted entities: {entities}")
                return entities
        except Exception as e:
            logger.error(f"Primary or Fallback NER failed: {e}")

        # Absolute fallback to rule-based if NER pipelines failed completely
        logger.info(f"[NER] All NER pipelines failed or skipped, using only rule-based NER.")
        return extract_entities_rule_based(query)


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
            type_priority = ["doctors", "departments", "rooms", "services", "buildings", "floors", "elevators", "opd", "ward", "office", "canteen"]

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
    

class InMemoryUserMemoryStore:
    def __init__(self):
        self.sessions = {}  # {user_id: ConversationMemory}
        self.lock = threading.Lock()

    def get(self, user_id):
        with self.lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = ConversationMemory()
            return self.sessions[user_id]

    def save(self, user_id, memory: ConversationMemory):
        with self.lock:
            self.sessions[user_id] = memory

    def clear(self, user_id):
        with self.lock:
            self.sessions.pop(user_id, None)

    def all_user_ids(self):
        return list(self.sessions.keys())


load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom in-memory user memory store
user_memory_store = InMemoryUserMemoryStore()

# Add CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration for Hospital Data ---
FAISS_INDEX_PATH = "rag_aimms_jammu"
HOSPITAL_MODEL_JSON_PATH = "hospital_building.json"
QA_PAIRS_JSON_PATH = "jammu_qa_pairs_cleaned.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALLOWED_FILES = {os.path.basename(HOSPITAL_MODEL_JSON_PATH),
                 os.path.basename(QA_PAIRS_JSON_PATH)}

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY not found in environment variables.")

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
        elif any(term in content_lower for term in ['appointment', 'booking', 'schedule']):
            metadata_payload["category"] = "appointments"
        elif any(term in content_lower for term in ['department', 'ward', 'unit', 'opd']):
            metadata_payload["category"] = "departments"
        elif any(term in content_lower for term in ['emergency', 'urgent', 'critical']):
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
        if question.lower().startswith(('what is', 'what are', 'what was', 'what were', 'what does', 'what do')):
            metadata_payload["priority"] = 1  # High priority for definition questions
        elif question.lower().startswith(('how to', 'how can', 'how do', 'how does', 'how is')):
            metadata_payload["priority"] = 2  # Medium priority for how-to questions
        elif question.lower().startswith(('where is', 'where are', 'where can', 'where do', 'where does')):
            metadata_payload["priority"] = 1  # High priority for location questions
        elif question.lower().startswith(('tell me', 'tell', 'provide', 'give', 'give me', 'provide me', 'i need', 'i want', ' i want to')):
            metadata_payload["priority"] = 2 # Mediam priority for genernal information questions
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
    def __init__(self):
        super().__init__()
        self.intent_labels = [
            "find_room", "find_department", "find_doctor", "find_service",
            "get_contact_info", "get_location", "get_directions", "get_schedule",
            "book_appointment", "emergency_info", "accessibility_info",
            "general_hospital_info", "definition_question", "how_to_question",
            "facility_inquiry", "medical_staff_inquiry", "department_inquiry"
        ]
    
    def classify_intent(self, query):
        # First try the parent classification
        intent = super().classify_intent(query)

        """Classify user intent based on query"""
        query_lower = query.lower()
        
        # Hospital-specific intents
        if any(term in query_lower for term in ['find', 'locate', 'where is', 'how can', 'tell', 'tell me', 'provide', 'location']) and any(term in query_lower for term in ['room', 'ward', 'opd', 'office']):
            return "find_room"
        elif any(term in query_lower for term in ['find', 'locate', 'where is', 'how can', 'provide', 'tell', 'tell me', ' i want to know', 'location', 'tell me about']) and any(term in query_lower for term in ['department', 'unit', 'block']):
            return "find_department"
        elif any(term in query_lower for term in ['find', 'locate', 'who is', 'how can', 'contact']) and any(term in query_lower for term in ['doctor', 'physician', 'specialist', 'professor', 'consultant']):
            return "find_doctor"
        elif any(term in query_lower for term in ['find', 'what', 'available', 'tell me', ' i want to know', 'tell me']) and any(term in query_lower for term in ['service', 'treatment', 'procedure']):
            return "find_service"
        elif any(term in query_lower for term in ['contact', 'phone', 'email', 'call', 'chat']):
            return "get_contact_info"
        elif any(term in query_lower for term in ['location', 'address', 'where is', 'directions']):
            return "get_location"
        elif any(term in query_lower for term in ['appointment', 'book', 'schedule', 'reserve', 'make a appointment']):
            return "get_schedule"
        
        return intent
    
# Update the global data loader initialization
data_loader = HospitalDataLoader()

embedding_models = {
    "multilingual": HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "qa": HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"),
    "general": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "ranking": HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5"),
    "hybrid": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"), 
    "factual": HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2"),
}
embedding = embedding_models["multilingual"]

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Adjusted chunk for potentially longer medical docs
    chunked_docs = text_splitter.split_documents(docs_for_faiss)

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
    bi_encoder_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    cross_encoder_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
except Exception as e:
    logger.error(f"Failed to load reranker models: {e}.")
    bi_encoder_reranker, cross_encoder_reranker = None, None

bm25_corpus_docs, bm25_tokenized_corpus = [], []
def initialize_bm25():
    global bm25_corpus_docs, bm25_tokenized_corpus
    documents = prepare_documents() # Uses hospital data now
    if not documents:
        logger.error("No documents available for BM25 initialization.")
        return None

    bm25_corpus_docs = documents
    bm25_tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

    if not bm25_tokenized_corpus:
        logger.error("BM25 tokenized corpus is empty.")
        return None

    logger.info(f"Initialized BM25 with {len(bm25_tokenized_corpus)} documents for hospital data.")
    return BM25Okapi(bm25_tokenized_corpus)
bm25 = initialize_bm25()

# Initialize NLU Processor
nlu_processor = NLUProcessor()

def get_embedding_model_for_query(query):
    query_lower = query.lower()

    # Hybrid model: Broad/general search or fuzzy context
    if any(word in query_lower for word in ["search", "find", "nearby", "available", "locate", "which room", "what rooms", "where can i find", "provide"]):
        return embedding_models["hybrid"]
    
    # QA-optimized: Direct factual questions
    if any(word in query_lower for word in ["what is", "who is", "where is", "define", "tell me about", "doctor", "department", "opd", 'office']):
        return embedding_models["qa"]
    
    # Factual: Explanation or in-depth info
    if any(word in query_lower for word in ["explain", "describe", "details about", "how does", "procedure", "treatment", "everything about"]):
        return embedding_models["factual"]
    
    # Ranking: Comparisons or list-based queries
    if any(word in query_lower for word in ["list all", "compare services", "rank doctors", "compare" "best", "vs", "versus", "list of some", "list of any", "list of five", "list of top", "table"]):
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
    - Joins fragments into full names
    - Deduplicates
    """
    from itertools import groupby

    def is_junk(token):
        return token.startswith("##") or len(token) <= 1 or not token[0].isalpha()

    cleaned_entities = {}
    for key, values in entities.items():
        new_vals = []
        current_phrase = []

        for token in values:
            token = token.replace("##", "").strip()

            if is_junk(token):
                continue

            # Group tokens into name phrases (e.g., ['Shruti', 'Sharma'])
            if token.istitle() or token[0].isupper():
                if current_phrase:
                    new_vals.append(" ".join(current_phrase))
                current_phrase = [token]
            else:
                current_phrase.append(token)

        if current_phrase:
            new_vals.append(" ".join(current_phrase))

        cleaned_entities[key] = sorted(set([x.strip() for x in new_vals if x.strip()]))

    return cleaned_entities


def refresh_faiss_and_bm25():
    global db_faiss, bm25, faiss_index_cache, data_loader
    logger.info("Refreshing FAISS index and BM25 model for hospital data.")

    # Reload data first
    #data_loader = HospitalDataLoader(hospital_filepath=HOSPITAL_MODEL_JSON_PATH, qa_filepath=QA_PAIRS_JSON_PATH) # Ensure it reloads hospital data

    faiss_index_cache.clear()
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True) 

    db_faiss = initialize_faiss() 
    bm25 = initialize_bm25() 

    if db_faiss and bm25:
        logger.info("FAISS and BM25 refreshed successfully for hospital data.")
    else:
        logger.error("Hospital data refresh failed for FAISS or BM25.")
    return db_faiss, bm25

def detect_query_complexity(query):
    query_lower = query.lower()
    if any(conj in query_lower for conj in ["and", "or", "but also", "as well as", "both" " all similar"]) and len(query.split()) > 7: 
        return "complex"
    if any(word in query_lower for word in ["list all services", "all doctors in department", "explain treatment options", "compare procedures", "compare", "list all", "list some"]): 
        return "complex"
    if len(query.split()) <= 5 and any(q_word in query_lower for q_word in ["where is", "dr. email", "phone for", "contact details of", "who is", "what is"]): 
        return "simple"
    return "normal"

def hybrid_retriever(query, k_simple=5, k_normal=8, k_complex=12):
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
    k_val = k_simple if complexity == "simple" else (k_normal if complexity == "normal" else k_complex)

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

def rerank_documents_bi_encoder(query, docs, top_k=8):
    if not bi_encoder_reranker:
        logger.warning("Bi-encoder reranker not available. Returning top documents without reranking.")
        return docs[:top_k]

    if not docs or len(docs) < 2:
        logger.info("Not enough documents for reranking. Returning as-is.")
        return docs[:top_k]

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = bi_encoder_reranker.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.info(f"Bi-encoder reranked {len(docs)} docs, returning top {top_k}.")
        return [doc for score, doc in scored_docs[:top_k]]
    except Exception as e:
        logger.error(f"Error during bi-encoder reranking: {e}. Returning top docs.")
        return docs[:top_k]

def rerank_documents_cross_encoder(query, docs, top_k=3):
    if not cross_encoder_reranker:
        logger.warning("Cross-encoder reranker not available. Returning top documents without reranking.")
        return docs[:top_k]

    if not docs or len(docs) < 2:
        logger.info("Not enough documents for reranking. Returning as-is.")
        return docs[:top_k]

    try:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder_reranker.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.info(f"Cross-encoder reranked {len(docs)} docs, returning top {top_k}.")
        return [doc for score, doc in scored_docs[:top_k]]
    except Exception as e:
        logger.error(f"Error during cross-encoder reranking: {e}. Returning top docs.")
        return docs[:top_k]

SYNONYM_MAP = {
    "elevator": ["lift"], "lift": ["elevator"],
    "toilet": ["washroom", "restroom", "lavatory", "wc"],
    "washroom": ["toilet", "restroom", "lavatory", "wc"],
    "restroom": ["toilet", "washroom", "lavatory", "wc"],
    "stairs": ["staircase"], "staircase": ["stairs"],
    "contactno": ["contact number", "phone no", "phone number", "contact"],
    "timings": ["operating hours", "open hours", "availability", "schedule"],
    "operating hours": ["timings", "open hours", "availability", "schedule"],
    "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0", "gf"],
    # Hospital specific
    "doctor": ["dr", "dr.", "physician", "consultant", "specialist", "professor"],
    "dr.": ["doctor", "physician", "consultant", "specialist", "professor"],
    "opd": ["outpatient department", "out-patient department", "clinic"],
    "er": ["emergency room", "emergency", "casualty"],
    "icu": ["intensive care unit"], "intensive care unit": ["icu"],
    "cardiology": ["heart department", "cardiac"],
    "neurology": ["neuro department", "nerve specialist"],
    "pediatrics": ["child care", "paediatrics", "peds"],
    "orthopedics": ["bone department", "ortho"],
    "radiology": ["x-ray department", "imaging"],
    "gynecology": ["women", "obgyn"], "women": ["gynecology"],
    "ophthalmology": ["eye"], "eye": ["ophthalmology"],
    "psychiatry": ["psychiatrist"], "psychiatrist": ["psychiatry"],
    "oncology": ["cancer"], "cancer": ["oncology"],
    "anesthesia": ["anesthesiology"], "anesthesiology": ["anesthesia"],
    "dermatology": ["skin"], "skin": ["dermatology"],
    "appointment": ["booking", "reservation", "appt"],
    "lab": ["laboratory"], "laboratory": ["lab"],
    "mri scan": ["mri"], "ct scan": ["ct"], "ultrasound scan": ["ultrasound"],
    "aiims": ["all india institute of medical sciences"], # Example hospital name
    # Equipment
    "ventilator machine": ["ventilator"], "ventilator": ["ventilator machine"],
    "wheel chair": ["wheelchair"], "wheelchair": ["wheel chair"]
    # Add more synonyms as needed
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
    hospital_matches = re.findall(r'\b(aiims(?:\s+\w+)?|jammu|\s+healthcare|city\s+hospital)\b', query_lower, re.IGNORECASE)
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

    # Lifts, Stairs, Washrooms (generic)
    if re.search(r'\b(lift|elevator)\b', query_lower): 
        entities["lifts"].append("lift")
    if re.search(r'\b(stairs|staircase)\b', query_lower): 
        entities["stairs"].append("stairs")
    if re.search(r'\b(washroom|toilet|restroom|lavatory|wc)\b', query_lower): 
        entities["washrooms"].append("washroom")

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


def detect_task_type_rule_based(query): # Adapted for hospital
    query_l = query.lower()
    if any(kw in query_l for kw in ["compare", "difference between", "vs ", "versus", "vs."]): return "compare"
    if any(kw in query_l for kw in ["book appointment", "appointment with", "reserve slot", "schedule visit", "want to meet", "book a slot", "fix appointment"]): return "booking_info" # For appointments
    if any(kw in query_l for kw in ["list all", "show all doctors", "all departments", "every service"]): return "listing_all"
    if any(kw in query_l for kw in ["list services", "list some", "list three", "overview of", "show some doctors", "some departments", "few treatments", "give examples of"]): return "listing_specific"
    if any(kw in query_l for kw in ["where is", "location of", "find near", "how to reach", "direction to", "which floor", "room number", "nearest", "reach the hospital", "hospital location"]): return "location"
    if any(kw in query_l for kw in ["email of", "contact for", "phone number", "call dr", "how to contact", "contact number", "mobile number", "hospital email", "reach dr" "website for hospital"]): return "contact_info"
    if any(kw in query_l for kw in ["how to", "explain", "procedure for", "what are symptoms", "symptoms of", "treatment for", "details about", "how is", "explanation", "tell me about", "elaborate", "how does it work"]): return "explanation"
    if any(kw in query_l for kw in ["operating hours", "timings", "when is opd open", "doctor schedule", "visiting hours", "opd hours", "working hours", "closing time"]): return "operating_hours" # or doctor_availability
    if any(kw in query_l for kw in ["doctor availability", "is dr available", "dr schedule", "dr schedule", "when is dr", "available doctor", "which doctor is available"]): return "doctor_availability"
    if any(kw in query_l for kw in ["department of", "info on", "tell me about cardiology", "cardiology services", "what is neurology", "specialty of", "department details", "which department handles"]): return "department_info"
    if any(kw in query_l for kw in ["service offered", "do you have", "available services", "is mri available", "ct scan facility", "x-ray available", "cost of service", "price of", "charges for", "ambulance available", "test cost"]): return "service_info"
    # Out of scope - keep general
    if any(kw in query_l for kw in ["weather", "time now", "news", "stock price", "meaning of life", "who are you", "what is your name", "are you humen"]): return "out_of_scope"
    return "general_information" # Default

import re

def extract_doctor_name(text: str) -> str:
    """
    Extracts a doctor's name like 'Dr Gaurav Gupta' or 'Dr. Shruti Sharma' from query.
    """
    try:
        match = re.search(r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        return match.group(0).strip() if match else ""
    except Exception as e:
        print(f"[extract_doctor_name error]: {e}")
        return ""

def format_doctor_response(doc):
    availability = format_doctor_availability(doc.get("availability", {}))
    return f"""
    Doctor: {doc['name']}
    Department: {doc.get('department')}
    Designation: {doc.get('designation')}
    Specialization: {doc.get('specialization')}
    Availability: {availability}
    Phone: {doc.get('phone') or 'Not available'}
    Email: {doc.get('email') or 'Not available'}
    Profile: {doc.get('profile') or 'N/A'}
""".strip()

def get_doctor_by_name(query_name, docs):
    """
    Looks for the doctor by name (normalized match) in retrieved documents' metadata.
    """
    norm = lambda x: x.lower().replace("dr ", "").strip()
    for doc in docs:
        for doctor in doc.metadata.get("doctor_profiles", []):
            if norm(query_name) in norm(doctor["name"]):
                return format_doctor_response(doctor)
    return None


from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    sym_spell.load_dictionary("resources/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
except Exception as e:
    logger.warning(f"Could not load spelling dictionary: {e}. Spelling correction might be affected.")


def correct_spelling(text, verbose=False):
    suggestions = sym_spell.lookup_compound(text.lower(), max_edit_distance=2)
    if suggestions:
        if verbose:
            print(f"[SpellCheck] Suggestions for '{text}':")
            for s in suggestions:
                print(f"  - {s.term} (distance: {s.distance}, count: {s.count})")
        corrected = suggestions[0].term
        if verbose:
            print(f"[SpellCheck] '{text}' → '{corrected}'")
        return corrected
    if verbose:
        print(f"[SpellCheck] No correction for '{text}'")
    return text


from rapidfuzz import fuzz
def collapse_repeated_letters(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1', text) # e.g. helllooo -> helo

def detect_conversational_intent(query):
    query_corrected_initial = correct_spelling(query) # Correct spelling first
    query_clean = collapse_repeated_letters(query_corrected_initial.lower().strip()) # Then normalize
    # logger.debug(f"[IntentDetection] Final cleaned query for convo intent: {query_clean} (from original: {query})")

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
    
    return None


def is_likely_room_code(token: str) -> bool: # General, can be kept
    return bool(re.match(r"^\d+[a-z]([-_\s]?\d+[a-z])?$", token, re.IGNORECASE)) or \
           bool(re.match(r"^[A-Za-z]?\d{2,}[A-Za-z]?$", token)) # e.g. R101, 303B

def normalize_room_code(token: str) -> str: # General, can be kept
    token = re.sub(r"[-_\s]+", "", token) # Remove separators: 3-a-2b -> 3a2b
    # Insert dash between digit and letter or letter and digit
    token = re.sub(r"(\d)([a-z])", r"\1-\2", token)
    token = re.sub(r"([a-z])(\d)", r"\1-\2", token)
    return token.upper()

def normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = collapse_repeated_letters(q) # helllo -> helo

    # Standardize medical/hospital suffixes or common terms
    q = q.replace("dept.", "department")
    q = q.replace("dr.", "doctor") # Standardize "dr." to "doctor" for easier matching later
    
    # Normalize room/OPD patterns like "opd3", "room 101a"
    q = re.sub(r"\b(opd|room|rm|cabin|ward|office|icu)\s*(\d+[a-z]?)", r"\1-\2", q, flags=re.IGNORECASE) # opd 3 -> opd-3
    q = re.sub(r"\b(\d+[a-z]?)\s*(opd|room|rm|cabin|ward|office|icu)", r"\2-\1", q, flags=re.IGNORECASE) # 3a opd -> opd-3a

    # Normalize lift lobby
    q = re.sub(r"lift\s*lobby[-\s]*(\d+)", r"lift lobby \1", q)

    # Collapse repeated characters (e.g., "heyyy" → "hey")
    q = re.sub(r"(.)\1{2,}", r"\1", q)

    # Tokenize and normalize room codes
    tokens = q.split()
    normalized_tokens = []
    for token in tokens:
        if is_likely_room_code(token):
            normalized_tokens.append(normalize_room_code(token))
        else:
            normalized_tokens.append(token)
    q = " ".join(normalized_tokens)

    # Remove unwanted punctuation (keeping '.', '-', and '@' for emails/domains)
    q = re.sub(r"[^\w\s\-\.@]", "", q)
    # Normalize spacing
    q = re.sub(r"\s+", " ", q).strip()
    return q

# Canonicalize entity value - uses SYNONYM_MAP, so it's fine.
def canonicalize_entity_value(entity_value):
    value_l = entity_value.lower().strip()
    for canonical, aliases in SYNONYM_MAP.items():
        all_forms = [canonical.lower()] + [a.lower() for a in aliases]
        if value_l in all_forms:
            # Return the main key of the synonym map (the canonical form)
            # Ensure the canonical form itself is returned if it matched one of its own aliases.
            # e.g. if SYNONYM_MAP has "opd": ["outpatient department"], and value_l is "outpatient department", return "opd".
            # For "doctor": ["dr", "dr."], if value_l is "dr.", return "doctor".
            # This assumes the key of SYNONYM_MAP is the desired canonical form.
            return canonical
    return entity_value # No match found, return original

def generate_clarification_suggestions(entities, memory): # General logic, adapt for hospital context
    suggestions = []
    recent_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=2)
    if entities.get("doctors") and not entities.get("departments"):
        for entity_info in recent_entities:
            if entity_info["type"] == "floors":
                suggestions.append(f"Could you specify which department Dr. {entities['doctors'][0]} belongs to?")
                break
    if entities.get("services") and not (entities.get("departments") or entities.get("rooms")):
        suggestions.append(f"Where is the {entities['services'][0]} service offered (e.g., department or room)?")
    
    if not suggestions and (entities.get("rooms") or entities.get("departments")):
        suggestions.append("Can you provide more details or rephrase your query?")
    elif not suggestions : # Generic fallbacks
        suggestions.extend(["Could you please provide more specific details?", "Can you rephrase your question?"])
    return suggestions[:2] # Limit suggestions

def classify_query_characteristics(query):
    query_l = query.lower()
    response_length = "short"
    if any(word in query_l for word in ["explain in detail", "everything about"]): response_length = "long"
    elif any(word in query_l for word in ["list", "summarize", "overview of"]): response_length = "medium"
    return {"response_length": response_length}

def detect_answer_style_and_tone(query):
    """
    Detects the preferred answer style (paragraph, bullet list, table) and tone (friendly, formal, professional)
    based on keywords in the query.
    """
    query_l = query.lower()
    # Default values
    style = "paragraph"
    tone = "professional_and_helpful"
    if any(word in query_l for word in ["bullet points", "list them", "in bullets", "pointwise", "give points", "give me points", "step by step", "in steps"]): style = "bullet_list"
    if any(word in query_l for word in ["in a table", "tabular format", "as a table", "table format", "make a table", "structured table"]): style = "table"
    if any(word in query_l for word in ["friendly", "casual", "informal", "talk like a friend", "light tone", "easy to understand", "simplify it"]): tone = "friendly_and_casual"
    if any(word in query_l for word in ["formal", "official statement", "strictly professional", "precise response", "in a formal tone", "business tone"]): tone = "formal_and_precise"
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


        pronoun_patterns = [
            r"\b(it)\b", r"\b(they)\b", r"\b(them)\b", 
            r"\b(their)\b(?!\s*(?:own|selves))", # their (but not their own, their selves)
            r"\b(his|her)\b",
            r"\b(its)\b",
            r"\b(this|that)\s*(one|department|doctor|room|service|place)?\b"
        ]
        
        for pattern in pronoun_patterns:
            match = re.search(pattern, rewritten_query, re.IGNORECASE)
            if match:
                pronoun = match.group(0)
                # Construct a meaningful replacement, e.g., "Dr. Smith's" for "his"
                replacement_text = salient_topic_entity_value
                if pronoun.lower() in ["his", "her", "their", "its"]:
                    replacement_text = f"{salient_topic_entity_value}'s"
                if not re.search(rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', query_lower_normalized, re.IGNORECASE):
                    rewritten_query = re.sub(pattern, replacement_text, rewritten_query, count=1, flags=re.IGNORECASE)
                    logger.info(f"[Coref Rewrite - Pronoun] Rewrote '{original_query}' → '{rewritten_query}' replacing '{pronoun}' with '{replacement_text}'")
                    break # Stop after first successful replacement

    if rewritten_query != original_query:
        logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
    return rewritten_query


def handle_small_talk(user_query, memory, user_id):
    import random
    convo_prompt = (
        f"You are a cheerful, friendly assistant at the Aiims building in Aiims Jammu.\n"
        f"The user is being casual, greeting, or saying goodbye.\n"
        f"Reply naturally and warmly. Add emojis occasionally. Never repeat yourself.\n"
        f"\nUser: {user_query}\nAssistant:"
    )

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.85)
        response = llm.invoke(convo_prompt).content.strip()
    except Exception as e:
        logger.error(f"Error in small talk handler: {e}")
        response = random.choice([
            "Hey there! 👋 How can I assist you at Aiims Jammu today?",
            "Hi! 😊 I'm here to help with anything related to the hospital.",
            "Hello! Just ask if you need directions, doctor info, or anything else!",
            "Hey! Need any help with appointments or departments?"
        ])

    # Initialize memory history if needed
    if not hasattr(memory, "history") or not isinstance(memory.history, list):
        memory.history = []

    memory.add_turn(user_query, response, extracted_entities_map={})  # optional: pass small talk entities

    # Save back to custom memory store
    user_memory_store.save(user_id, memory)
    return {"answer": response}


def chat(user_query: str, user_id: str):
    request_start_time = datetime.now()
    logger.info(f"--- New Chat Request (Hospital) --- User ID: {user_id} | Query: '{user_query}'")

    # Get existing memory or new one from your custom memory store
    conv_memory = user_memory_store.get(user_id)


    original_user_query = user_query.strip()
    query_lower_raw = original_user_query.lower()
    # Early check for conversational intents that might not need full processing pipeline
    # Detect conversational intent BEFORE translation or complex normalization
    convo_intent = detect_conversational_intent(original_user_query) 
    hospital_entity_keywords = [
        "room", "opd", "icu", "ward", "doctor", "dr", "nurse", "staff", "physician", "consultant", "specialist",
        "department", "cardiology", "neurology", "oncology", "pediatrics", "radiology", "surgery", "clinic",
        "service", "x-ray", "mri", "scan", "test", "appointment", "treatment", "procedure",
        "hospital", "aiims", "building", "floor", "location", "find", "where", "contact", "phone", "email",
        "availability", "schedule", "hours", "timings"
    ]
    
    # If conversational intent detected AND query doesn't seem to contain substantial entity keywords, handle as small talk
    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(keyword in query_lower_raw for keyword in hospital_entity_keywords):
             # Pass original query to small talk handler to preserve nuance
             return handle_small_talk(original_user_query, conv_memory, user_id)


    # Proceed with full pipeline for substantive queries or mixed conversational/substantive queries
    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
    translated_query, detected_input_lang = detect_and_translate(cleaned_query_for_lang_detect)
    processed_query = (
        rewrite_query_with_memory(translated_query, conv_memory) 
        if len(translated_query.split()) < 7 and not any(translated_query.lower().startswith(q_word) for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"]) 
        else translated_query
    )

    # NLU processing on the potentially rewritten and normalized query
    extracted_query_entities = nlu_processor.extract_entities(processed_query)
    extracted_query_entities = clean_extracted_entities(extracted_query_entities)

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

    
    hospital_entity_keywords = [
        "room", "opd", "icu", "ward", "doctor", "dr", "nurse", "staff", "physician", "consultant", "specialist",
        "department", "cardiology", "neurology", "oncology", "pediatrics", "radiology", "surgery", "clinic",
        "service", "x-ray", "mri", "scan", "test", "appointment", "treatment", "procedure",
        "hospital", "aiims", "building", "floor", "location", "find", "where", "contact", "phone", "email",
        "availability", "schedule", "hours", "timings"
    ]
    query_lower_raw = original_user_query.lower()


    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(word in query_lower_raw for word in hospital_entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, user_id)
        else:
            # Treat as mixed: greeting + actual question → combine both
            greeting_resp = handle_small_talk(original_user_query, conv_memory, user_id)["answer"]
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

    retrieved_docs = hybrid_retriever(processed_query, k_simple=6, k_normal=10, k_complex=15) # Increased k for potentially more complex data
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

    bi_reranked_docs = rerank_documents_bi_encoder(processed_query, retrieved_docs, top_k=6) # Increase bi-encoder top_k
    final_docs_for_llm = rerank_documents_cross_encoder(processed_query, bi_reranked_docs, top_k=3) # Increase cross-encoder top_k

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
    for term in entity_terms_to_check:
        found_in_docs = any(term in doc.page_content.lower() for doc in final_docs_for_llm)
        if not found_in_docs:
            missing_entities.append(term)

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
    if not final_docs_for_llm and retrieved_docs: 
        final_docs_for_llm = retrieved_docs[:3]; 
        logger.warning("Reranking resulted in zero documents. Using top 3 from initial hybrid retrieval for LLM.")

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
    if response_length_hint == "long" or task_type in ["explanation", "comparison", "listing_all"] or "complex" in detect_query_complexity(processed_query) :
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.4 # Slightly lower temp for factual long
    elif task_type in ["contact_info", "location", "doctor_availability"] and response_length_hint == "short":
        groq_llm_model_name, temperature_val = "llama3-8b-8192", 0.15 # Precise for short facts
    else: 
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.25 # Default to more capable model with moderate temp
    logger.info(f"Using Groq model (hospital): {groq_llm_model_name} with temperature: {temperature_val}")
    
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

    user_memory_store.save(user_id, conv_memory)

    # Translate response if needed
    try:
        if target_lang_code and target_lang_code != "en": 
            final_response_text = GoogleTranslator(source="en", target=target_lang_code).translate(final_response_text)
            logger.info(f"Translated response to {target_lang_code}.")
        elif detected_input_lang != "en" and detected_input_lang is not None: # check detected_input_lang is not None
            final_response_text = GoogleTranslator(source="en", target=detected_input_lang).translate(final_response_text)
            logger.info(f"Translated response back to input language {detected_input_lang}.")
    except Exception as e: 
        logger.warning(f"Failed to translate final response: {e}")
    
    processing_time = (datetime.now() - request_start_time).total_seconds()
    logger.info(f"--- Chat Request Completed (Hospital) --- Time: {processing_time:.2f}s")
    debug_info = {
        "detected_task_type": task_type,
        "processed_query": processed_query,
        "detected_input_lang": detected_input_lang,
        "target_response_lang": target_lang_code,
        "answer_style": answer_style,
        "answer_tone": answer_tone,
        "response_length_hint": response_length_hint,
        "llm_model_used": groq_llm_model_name,
        "retrieved_docs_count_initial": len(retrieved_docs) if retrieved_docs else 0,
        "retrieved_docs_count_final_llm": len(final_docs_for_llm) if final_docs_for_llm else 0,
        "final_doc_ids_for_llm": [doc.metadata.get("source_doc_id","Unknown") for doc in final_docs_for_llm] if final_docs_for_llm else [],
        "processing_time_seconds": round(processing_time, 2),
        "conversational_intent": convo_intent,
        "extracted_entities": extracted_query_entities,
        "query_complexity": classify_query_characteristics(processed_query),
        "missing_entities_in_docs": missing_entities
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
    logger.info("Hospital data and retrieval models refreshed successfully.")
    return {"message": "Hospital data and retrieval models refreshed successfully."}


@app.get("/api/metadata-tags")
async def get_metadata_tags():
    tags = data_loader.get_all_metadata_tags()
    return {"tags": tags}


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