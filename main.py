from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
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
from sentence_transformers import CrossEncoder
from flask_session import Session
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import torch
from collections import Counter
import sys
import io

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
            # Intent classification using facebook/bart-large-mnli
            self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            self.intent_labels = [
                "location", "contact_info", "booking_info", "explanation", "comparison", # booking_info can be for appointments
                "operating_hours", "listing_all", "listing_specific", "out_of_scope", "general_information",
                "doctor_availability", "department_info", "service_info" # Added more specific intents
            ]
            # Primary NER model
            self.ner_pipeline_primary = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
            
            # Secondary fallback NER model (XLM-RoBERTa multilingual)
            self.ner_pipeline_fallback = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
            
            logger.info("NLU models initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize NLU models: {e}")
            self.intent_classifier = None
            self.ner_pipeline_primary = None
            self.ner_pipeline_fallback = None

    def classify_intent(self, query):
        if not self.intent_classifier:
            logger.warning("Intent classifier not available. Falling back to rule-based detection.")
            return detect_task_type_rule_based(query)
        try:
            result = self.intent_classifier(query, self.intent_labels, multi_label=False)
            intent = result["labels"][0]
            confidence = result["scores"][0]
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            return intent
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
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

                if not value or value.startswith("##"): continue


                if entity_type == "LOC":
                    # Try to be more specific based on keywords
                    if "hospital" in value.lower() or "aiims" in value.lower() : # Assuming AIIMS is a hospital name pattern
                        entities["hospitals"].append(value)
                    elif "building" in value.lower() or "diagnostic" in value.lower() or "wing" in value.lower() or "block" in value.lower(): # diagnostic, wing are from example
                        entities["buildings"].append(value)
                    elif "floor" in value.lower() or re.match(r'\d+', value.split()[0] if value else ""): # Check if first part is a number for floor
                        entities["floors"].append(value)
                    elif "room" in value.lower() or re.match(r'\w*\d+\w*', value): # Simple room pattern
                        entities["rooms"].append(value)
                    elif "lift" in value.lower():
                        entities["lifts"].append(value)
                    elif "stair" in value.lower():
                        entities["stairs"].append(value)
                    elif "washroom" in value.lower() or "restroom" in value.lower() or "toilet" in value.lower():
                        entities["washrooms"].append(value)
                    else:
                        entities["general_terms"].append(value) # Default for LOC if not specific
                elif entity_type == "ORG": # Could be hospital or department
                    if "hospital" in value.lower() or "aiims" in value.lower():
                         entities["hospitals"].append(value)
                    elif any(dept_kw in value.lower() for dept_kw in ["department", "cardiology", "opd", "clinic", "radiology", "surgery"]): # Add more KWs
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
                if not any(entities.get(k) for k in ["doctors", "departments", "rooms", "hospitals"]):
                    logger.info(f"[NER] Still sparse, augmenting with rule-based NER.")
                    rule_based_entities = extract_entities_rule_based(query)
                    for key, value_list in rule_based_entities.items():
                        if value_list and not entities.get(key):
                            entities[key] = value_list
                        elif value_list:
                             entities[key] = sorted(list(set(entities[key] + value_list)))
                
                logger.info(f"[NER Final] Extracted entities: {entities}")
                return entities
        except Exception as e:
            logger.error(f"Primary or Fallback NER failed: {e}")

        # Absolute fallback to rule-based if NER pipelines failed completely
        logger.info(f"[NER] All NER pipelines failed or skipped, using only rule-based NER.")
        return extract_entities_rule_based(query)


# === CONVERSATION MEMORY MODULE ===
class ConversationMemory:
    def __init__(self, max_history_turns=12, summary_threshold=8, llm=None):
        self.history = []
        self.contextual_entities = []
        self.current_topic = None
        self.last_entity_by_type = {}
        self.max_history_turns = max_history_turns
        self.summary_threshold = summary_threshold
        self.summary = ""  # New field for summary
        self.llm = llm or ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.2)

    def summarize_history(self):
        """Summarizes the conversation history to keep the context concise."""
        if not self.llm:
            logger.warning("LLM not provided to ConversationMemory. Cannot summarize history.")
            return

        # Select turns to summarize, excluding the most recent ones
        turns_to_summarize = self.history[:-self.summary_threshold // 2]
        if len(turns_to_summarize) < 4:  # Don't summarize very short histories
            return

        history_text = ""
        for turn in turns_to_summarize:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = f"""Concisely summarize the key entities, topics, and user intents from this conversation.
        Focus on information that would be critical for understanding future related questions.
        Do not mention specific turn numbers.

        Conversation:
        {history_text}

        Concise Summary:"""

        try:
            logger.info(f"Summarizing {len(turns_to_summarize)} turns of conversation history.")
            summary_response = self.llm.invoke(prompt)
            self.summary = summary_response.content.strip()
            # Replace the summarized turns with a placeholder
            self.history = self.history[-self.summary_threshold // 2:]
            logger.info(f"New conversation summary: {self.summary}")
        except Exception as e:
            logger.error(f"Error during conversation summarization: {e}")

    def add_turn(self, user_query, assistant_response, extracted_entities_map):
        turn_index = len(self.history)
        if self.summary: # Adjust turn index if history was summarized
            turn_index += self.max_history_turns - len(self.history)

        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "turn_index": turn_index
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
                        self.last_entity_by_type[entity_type] = entity_value
            if self.contextual_entities:
                self.current_topic = self.contextual_entities[-1]

        # Prune history and summarize if it exceeds limits
        if len(self.history) > self.max_history_turns:
            self.summarize_history()
            # Ensure history doesn't exceed max_history_turns even after summarization attempt
            self.history = self.history[-self.max_history_turns:]


    def get_last_entity_by_priority(self, type_priority=None):
        if type_priority is None:
            type_priority = ["doctors", "departments", "services", "rooms", "floors", "hospitals", "buildings"]
        for entity_type in type_priority:
            if entity_type in self.last_entity_by_type:
                return self.last_entity_by_type[entity_type]
        return None

    def get_contextual_history_text(self, num_turns=5):
        history_text = ""
        if self.summary:
            history_text += f"Conversation Summary:\n{self.summary}\n\nRecent History:\n"

        recent_turns = self.history[-num_turns:] if isinstance(self.history, list) else []
        for i, turn in enumerate(recent_turns):
            turn_index = turn.get("turn_index", "N/A")
            user_msg = turn.get("user", "[no user input]")
            assistant_msg = turn.get("assistant", "[no assistant response]")
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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey_for_hospital_app_v1") # Changed key name
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session_hospital/" # Changed session dir
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

def ensure_list(val):
    if isinstance(val, list): return val
    elif val: return [val]
    return []

# --- Configuration for Hospital Data ---
FAISS_INDEX_PATH = "rag_hospital_data_index" # New FAISS index path
HOSPITAL_MODEL_JSON_PATH = "Hospital_building.json" 
ALLOWED_FILES = {os.path.basename(HOSPITAL_MODEL_JSON_PATH)}
# --- End Configuration ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY not found in environment variables.")

class RNIDataLoader: # Renaming to GenericDataLoader or HospitalDataLoader might be better
    def __init__(self, filepath=HOSPITAL_MODEL_JSON_PATH): # Default to hospital data
        self.filepath = filepath
        self.building_data = self.load_json_secure(self.filepath) # building_data now holds hospital data
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
        if not self.building_data: # building_data is now list of hospital entries
            return {
                "rooms": [], "departments": [], "doctors": [], "services": [],
                "lifts": [], "stairs": [], "emergencyExits": [], "entrances": [],
                "hospitals": [], "buildings": []
            }

        rooms_set, departments_set, doctors_set, services_set = set(), set(), set(), set()
        lifts_set, stairs_set, exits_set, entrances_set = set(), set(), set(), set()
        hospitals_set, buildings_set = set(), set()

        for item in self.building_data: # item is a hospital/room entry from the list
            hospital_name = item.get("hospitalName")
            if hospital_name: hospitals_set.add(hospital_name)

            loc_context = item.get("locationContext", {})
            building_name = loc_context.get("buildingName")
            if building_name: buildings_set.add(building_name)

            room_details = item.get("roomDetails", {})
            if room_details.get("roomName"): rooms_set.add(room_details["roomName"])
            if room_details.get("roomNumber"): rooms_set.add(str(room_details["roomNumber"]))

            for dept in item.get("departments", []):
                if dept.get("departmentName"): departments_set.add(dept["departmentName"])
                for service_name in ensure_list(dept.get("relatedServices", [])): # Services from departments
                    services_set.add(service_name)
                for doctor in dept.get("doctors", []):
                    if doctor.get("name"): doctors_set.add(doctor["name"])

            for service_item in item.get("servicesOffered", []): # Top-level services in the item
                if service_item.get("serviceName"): services_set.add(service_item["serviceName"])
            
            access = item.get("accessibility", {}).get("nearestAccessPoints", {})
            for ap_group, collector_set in [("lifts", lifts_set), ("stairs", stairs_set), 
                                            ("emergencyExits", exits_set), ("entrances", entrances_set)]:
                for ap_entry in ensure_list(access.get(ap_group, [])):
                    if ap_entry.get("name"): collector_set.add(ap_entry["name"])
        
        return {
            "rooms": sorted(list(rooms_set)),
            "departments": sorted(list(departments_set)),
            "doctors": sorted(list(doctors_set)),
            "services": sorted(list(services_set)),
            "lifts": sorted(list(lifts_set)),
            "stairs": sorted(list(stairs_set)),
            "emergencyExits": sorted(list(exits_set)),
            "entrances": sorted(list(entrances_set)),
            "hospitals": sorted(list(hospitals_set)),
            "buildings": sorted(list(buildings_set))
        }
    
    def get_all_metadata_tags(self):
        # Hospital_model.json doesn't have a direct "metadata.tags" field.
        # This function would need to be adapted to synthesize tags from other fields
        # (e.g., room types, services, department names, accessibility features) if needed.
        # For now, returning empty list to avoid errors.
        logger.warning("get_all_metadata_tags: 'metadata.tags' field not directly available in Hospital_model.json. Returning empty list.")
        return []

    def get_metadata_tag_counts(self):
        # Similar to get_all_metadata_tags, this needs adaptation for Hospital_model.json.
        logger.warning("get_metadata_tag_counts: 'metadata.tags' field not directly available in Hospital_model.json. Returning empty dict.")
        return {}


data_loader = RNIDataLoader(filepath=HOSPITAL_MODEL_JSON_PATH) # Explicitly use new path

def format_operating_hours(hours_data): # Kept for general use, might not be directly used by Hospital_model.json as is
    if not hours_data: return "N/A"
    order = ["mondayToFriday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
    parts = []
    for day_key in order:
        if day_key not in hours_data: continue
        label = day_key.replace("mondayToFriday", "Mon–Fri").replace("To", "–").capitalize()
        time_info = hours_data[day_key]
        if isinstance(time_info, dict): # Handles {"start": "9am", "end": "5pm"}
            parts.append(f"{label}: {time_info.get('start', 'N/A')}–{time_info.get('end', 'N/A')}")
        elif isinstance(time_info, str): # Handles "9am-5pm"
            parts.append(f"{label}: {time_info}")
        else: # Fallback for other types
             parts.append(f"{label}: {str(time_info)}")
    return "; ".join(parts) if parts else "N/A"

def format_doctor_availability(availability_data):
    if not availability_data:
        return "Not specified"
    days = ensure_list(availability_data.get("days", []))
    time = availability_data.get("time", "Not specified")
    if not days:
        return f"Time: {time}" if time != "Not specified" else "Availability: Not specified"
    return f"Days: {', '.join(days)}; Time: {time}"

# format_response_channels is likely not directly applicable.
# Contact info will be part of the document content from various fields.

def prepare_documents():
    if not data_loader.building_data: # This is now hospital data
        logger.error("Hospital data not loaded. Cannot prepare documents.")
        return []

    documents = []
    # Hospital_model.json is a list of items, each can be treated as a document source
    for item_index, item_data in enumerate(data_loader.building_data):
        content_parts = []
        # Unique ID for the document, combining hospital and room/location ID
        room_dt = item_data.get("roomDetails", {})
        source_id = f"{item_data.get('hospitalId', 'unknown_hospital')}_{room_dt.get('locationId', f'item_{item_index}')}"
        
        metadata_payload = {
            "source_doc_id": source_id,
            "hospital_id": item_data.get("hospitalId"),
            "hospital_name": item_data.get("hospitalName"),
            "hospital_type": item_data.get("hospitalType"),
        }

        content_parts.append(f"Hospital: {item_data.get('hospitalName', 'N/A')} ({item_data.get('hospitalType', 'N/A')}).")

        # Location Context
        loc_ctx = item_data.get("locationContext", {})
        content_parts.append(
            f"Location Context: Venue '{loc_ctx.get('venueName', 'N/A')}', Building '{loc_ctx.get('buildingName', 'N/A')}', "
            f"Zone '{loc_ctx.get('zone', 'N/A')}', Area Type '{loc_ctx.get('areaType', 'N/A')}', Indoor: {loc_ctx.get('indoor', 'N/A')}."
        )
        metadata_payload.update({
            "venue_name": loc_ctx.get("venueName"),
            "building_name": loc_ctx.get("buildingName"),
            "zone": loc_ctx.get("zone"),
        })

        # Room Details
        if room_dt:
            content_parts.append(
                f"Room Details: Name '{room_dt.get('roomName', 'N/A')}', Number '{room_dt.get('roomNumber', 'N/A')}', "
                f"Floor {room_dt.get('floor', 'N/A')}, Type '{room_dt.get('roomType', 'N/A')}', SubType '{room_dt.get('roomSubType', 'N/A')}'. "
                f"Function: {room_dt.get('roomFunction', 'General Purpose')}."
            )
            metadata_payload.update({
                "room_location_id": room_dt.get("locationId"),
                "room_name": room_dt.get("roomName"),
                "room_number": str(room_dt.get("roomNumber", "")),
                "floor": str(room_dt.get("floor", "")),
                "room_type": room_dt.get("roomType"),
                "room_subtype": room_dt.get("roomSubType"),
            })

            geo = room_dt.get("geometry", {}).get("geo", {})
            cartesian = room_dt.get("geometry", {}).get("cartesian", {})
            door_coords = cartesian.get("door", {})
            content_parts.append(
                f"Coordinates: Geo(Lat {geo.get('latitude', 'N/A')}, Lon {geo.get('longitude', 'N/A')}), "
                f"Cartesian(X {cartesian.get('x', 'N/A')}, Y {cartesian.get('y', 'N/A')}), "
                f"Door(X {door_coords.get('x', 'N/A')}, Y {door_coords.get('y', 'N/A')})."
            )
            if geo: metadata_payload["geo_coordinates"] = f"{geo.get('latitude')},{geo.get('longitude')}"

            door_info = room_dt.get("door", {})
            content_parts.append(
                f"Door: Type '{door_info.get('type', 'N/A')}', Mech '{door_info.get('mechanism', 'N/A')}', "
                f"Motion '{door_info.get('motion', 'N/A')}'. SmartLock: {door_info.get('smartLock', False)}."
            )
            utils = room_dt.get("utilities", {})
            utils_str = ", ".join(f"{k}: {v}" for k, v in utils.items()) if utils else "N/A"
            content_parts.append(f"Room Utilities: {utils_str}.")

        # Hospital Contact Information
        contact_info = item_data.get("contactInformation", {})
        address = contact_info.get("address", {})
        content_parts.append(
            f"Hospital Contact: Phone(s) '{contact_info.get('phoneNumbers', 'N/A')}', Email '{contact_info.get('email', 'N/A')}'. "
            f"Address: {address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')}, {address.get('postalCode', '')}, {address.get('country', '')}."
        )
        metadata_payload.update({
            "hospital_phone": contact_info.get("phoneNumbers"),
            "hospital_email": contact_info.get("email"),
            "hospital_address_city": address.get("city"),
        })
        
        # Services Offered (at this item level)
        services_offered_list = []
        for service in ensure_list(item_data.get("servicesOffered", [])):
            svc_desc = (f"Service: {service.get('serviceName', 'N/A')} (ID: {service.get('serviceId', 'N/A')}). "
                        f"Desc: {service.get('description', 'N/A')}. Dept: {service.get('department', 'N/A')}. "
                        f"Specialties: {', '.join(ensure_list(service.get('specialties', [])))}.")
            content_parts.append(svc_desc)
            services_offered_list.append(service.get('serviceName'))
        if services_offered_list:
            metadata_payload["services_directly_offered"] = sorted(list(set(services_offered_list)))


        # Departments and Doctors (associated with this item)
        department_names_list = []
        doctor_names_list = []
        related_services_list = []

        for dept in ensure_list(item_data.get("departments", [])):
            dept_name = dept.get('departmentName', 'N/A')
            department_names_list.append(dept_name)
            dept_str = (f"Department: {dept_name} (ID: {dept.get('departmentId', 'N/A')}). "
                        f"Floor: {dept.get('floor', 'N/A')}. Contact: {dept.get('contactNumber', 'N/A')}. ")
            
            current_dept_services = ensure_list(dept.get("relatedServices", []))
            related_services_list.extend(current_dept_services)
            if current_dept_services:
                dept_str += f"Related Services: {', '.join(current_dept_services)}. "
            content_parts.append(dept_str)

            for doctor in ensure_list(dept.get("doctors", [])):
                doc_name = doctor.get('name', 'N/A Doctor')
                doctor_names_list.append(doc_name)
                doc_avail = format_doctor_availability(doctor.get("availability", {}))
                doc_str = (f"Doctor: {doc_name} (ID: {doctor.get('doctorId', 'N/A')}). "
                           f"Specialization: {doctor.get('specialization', 'N/A')}, Designation: {doctor.get('designation', 'N/A')}. "
                           f"Email: {doctor.get('email', 'N/A')}, Phone: {doctor.get('phone', 'N/A')}. "
                           f"Availability: {doc_avail}. Profile: {doctor.get('profile','N/A')}. "
                           f"About: {doctor.get('about', 'N/A')}")
                content_parts.append(doc_str)
        
        if department_names_list: metadata_payload["associated_departments"] = sorted(list(set(department_names_list)))
        if doctor_names_list: metadata_payload["associated_doctors"] = sorted(list(set(doctor_names_list)))
        if related_services_list: metadata_payload["department_related_services"] = sorted(list(set(related_services_list)))


        # Accessibility
        acc = item_data.get("accessibility", {})
        content_parts.append(f"Wheelchair Accessible: {acc.get('isWheelchairAccessible', False)}.")
        acc_features = ", ".join(ensure_list(acc.get("features", []))) or "N/A"
        content_parts.append(f"Accessibility Features: {acc_features}.")
        metadata_payload["accessibility_features_summary"] = acc_features[:150] # Increased summary length

        ap = acc.get("nearestAccessPoints", {})
        ap_parts = []
        for ap_type, ap_list_of_obj in ap.items(): # e.g. "lifts": [{name:"Lift-1", distance:10}, ...]
            for ap_item in ensure_list(ap_list_of_obj):
                ap_parts.append(
                    f"{ap_item.get('name', f'Unnamed {ap_type.singularize() if hasattr(ap_type, isinstance) and isinstance(ap_type, str) else ap_type}')} " # simple singularize attempt
                    f"({ap_item.get('distanceMeters', 'N/A')}m)"
                )
        content_parts.append(f"Nearest Access Points: {', '.join(ap_parts) if ap_parts else 'N/A'}.")

        amenities = ", ".join(ensure_list(acc.get("amenities", []))) or "N/A"
        content_parts.append(f"Amenities: {amenities}.")
        metadata_payload["amenities_summary"] = amenities[:150]

        # Emergency Services (boolean)
        content_parts.append(f"Emergency Services Available (in this context/hospital): {item_data.get('emergencyServices', False)}.")
        metadata_payload["emergency_services_flag"] = item_data.get('emergencyServices', False)

        # Add schema info for context
        # content_parts.append(f"Schema Version: {item_data.get('version', 'N/A')}, Schema ID: {item_data.get('$id', 'N/A')}.")

        # Final Document
        page_content = "\n".join(filter(None, content_parts))
        documents.append(Document(page_content=page_content, metadata=metadata_payload))

    logger.info(f"Prepared {len(documents)} documents from hospital data for FAISS index.")
    return documents

embedding_models = {
    "multilingual": HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "qa": HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"),
    "general": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "ranking": HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5"),
    "hybrid": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"), # Good general purpose, multilingual
    "factual": HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2"),
}
embedding = embedding_models["hybrid"] # Changed default to hybrid

faiss_index_cache = {}
def initialize_faiss(embedding_model_instance=None):
    global faiss_index_cache
    effective_embedding_model = embedding_model_instance if embedding_model_instance else embedding
    model_id = effective_embedding_model.model_name # Ensure model_name attribute exists

    # Robust way to get model identifier if model_name is not directly on the instance
    if not hasattr(effective_embedding_model, 'model_name') and hasattr(effective_embedding_model, 'client') and hasattr(effective_embedding_model.client, 'model'):
        model_id = effective_embedding_model.client.model # Fallback for some HuggingFaceEmbeddings structures
    elif not hasattr(effective_embedding_model, 'model_name'):
        # Fallback if model_name cannot be determined, use a generic key or hash
        model_id = "unknown_model_" + str(type(effective_embedding_model))


    if model_id in faiss_index_cache:
        logger.info(f"Using cached FAISS index for model: {model_id}")
        return faiss_index_cache[model_id]

    # Ensure FAISS_INDEX_PATH directory exists before attempting to load or save
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
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
            if os.path.exists(FAISS_INDEX_PATH): # Should already exist due to makedirs
                # Clean up potentially corrupted files before rebuilding
                if os.path.exists(faiss_file): os.remove(faiss_file)
                if os.path.exists(pkl_file): os.remove(pkl_file)
            # os.makedirs(FAISS_INDEX_PATH, exist_ok=True) # Already done

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
nlu_processor = NLUProcessor() # Uses updated NLUProcessor logic

def get_embedding_model_for_query(query):
    query_lower = query.lower()
    # Simplified routing, can be expanded
    if any(word in query_lower for word in ["what is", "who is", "where is", "define", "tell me about", "doctor", "department"]):
        return embedding_models["qa"]
    if any(word in query_lower for word in ["explain", "describe", "details about", "how does", "procedure", "treatment"]):
        return embedding_models["factual"]
    if any(word in query_lower for word in ["list all", "compare services", "rank doctors"]): # Adjusted examples
        return embedding_models["ranking"]
    # Hybrid model for broad/general search or fuzzy context, also good default
    # Defaulting to hybrid as it's robust
    model_to_use = embedding_models.get("hybrid", embedding_models["multilingual"]) # Ensure hybrid exists
    logger.info(f"[Embedding Model Routing] Using: {getattr(model_to_use, 'model_name', 'N/A')} for query: {query}")
    return model_to_use


def refresh_faiss_and_bm25():
    global db_faiss, bm25, faiss_index_cache, data_loader
    logger.info("Refreshing FAISS index and BM25 model for hospital data.")

    # Reload data first
    data_loader = RNIDataLoader(filepath=HOSPITAL_MODEL_JSON_PATH) # Ensure it reloads hospital data

    faiss_index_cache.clear()
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    # os.makedirs(FAISS_INDEX_PATH, exist_ok=True) # initialize_faiss will handle this

    db_faiss = initialize_faiss() # This will call prepare_documents internally
    bm25 = initialize_bm25()   # This will also call prepare_documents

    if db_faiss and bm25:
        logger.info("FAISS and BM25 refreshed successfully for hospital data.")
    else:
        logger.error("Hospital data refresh failed for FAISS or BM25.")
    return db_faiss, bm25

def detect_query_complexity(query): # General logic, should still apply
    query_lower = query.lower()
    if any(conj in query_lower for conj in [" and ", " or ", " but also ", " as well as "]) and len(query.split()) > 7: return "complex"
    if any(word in query_lower for word in ["list all services", "all doctors in department", "explain treatment options", "compare procedures"]): return "complex"
    if len(query.split()) <= 5 and any(q_word in query_lower for q_word in ["where is room", "dr. email", "phone for cardiology"]): return "simple"
    return "normal"

def hybrid_retriever(query, k_simple=5, k_normal=8, k_complex=12):
    selected_embedding_instance = get_embedding_model_for_query(query)
    
    # Get model_id safely
    model_id = getattr(selected_embedding_instance, 'model_name', None)
    if not model_id and hasattr(selected_embedding_instance, 'client') and hasattr(selected_embedding_instance.client, 'model'):
        model_id = selected_embedding_instance.client.model
    if not model_id: model_id = "unknown_model_hybrid_retriever"


    current_db_faiss = (
        faiss_index_cache.get(model_id) # Use determined model_id
        or initialize_faiss(selected_embedding_instance)
    )

    if not current_db_faiss:
        logger.error("FAISS database (hospital) not available.")
        return []

    if not bm25:
        logger.error("BM25 (hospital) not available. Falling back to FAISS only.")
        # Ensure as_retriever exists and is callable
        if hasattr(current_db_faiss, 'as_retriever') and callable(current_db_faiss.as_retriever):
            return current_db_faiss.as_retriever(search_kwargs={"k": k_normal}).get_relevant_documents(query)
        else: # Fallback if as_retriever is not available (e.g. wrong object type)
            logger.error("current_db_faiss object does not have as_retriever method.")
            # Attempt a similarity search if available
            try:
                return current_db_faiss.similarity_search(query, k=k_normal)
            except Exception as e_sim:
                logger.error(f"FAISS similarity_search also failed: {e_sim}")
                return []


    complexity = detect_query_complexity(query)
    k_val = k_simple if complexity == "simple" else (k_normal if complexity == "normal" else k_complex)

    logger.info(f"Hybrid retrieval (hospital) for '{query}' → complexity: {complexity}, k={k_val}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Ensure as_retriever exists before calling
        if hasattr(current_db_faiss, 'as_retriever') and callable(current_db_faiss.as_retriever):
             faiss_future = executor.submit(
                current_db_faiss.as_retriever(search_kwargs={"k": k_val}).get_relevant_documents, query
            )
        else: # Fallback if no as_retriever
            logger.warning("FAISS object does not have as_retriever. Submitting similarity_search to executor.")
            try:
                faiss_future = executor.submit(current_db_faiss.similarity_search, query, k=k_val)
            except Exception as e_ss_submit:
                 logger.error(f"Failed to submit FAISS similarity_search: {e_ss_submit}")
                 # Create a future that returns an empty list immediately
                 from concurrent.futures import Future
                 faiss_future = Future()
                 faiss_future.set_result([])


        bm25_future = executor.submit(bm25_retriever_func, query, k_val)

        try:
            faiss_docs = faiss_future.result(timeout=10)
            bm25_top_docs = bm25_future.result(timeout=10)
        except TimeoutError:
            logger.warning("Retrieval timed out (hospital).")
            faiss_docs = faiss_future.result() if faiss_future.done() and not faiss_future.cancelled() else []
            bm25_top_docs = bm25_future.result() if bm25_future.done() and not bm25_future.cancelled() else []
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
    "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0", "gf"],
    # Hospital specific
    "doctor": ["dr", "dr.", "physician", "consultant", "specialist"],
    "dr.": ["doctor", "physician", "consultant", "specialist"],
    "opd": ["outpatient department", "out-patient department", "clinic"],
    "er": ["emergency room", "emergency", "casualty"],
    "icu": ["intensive care unit"],
    "cardiology": ["heart department", "cardiac"],
    "neurology": ["neuro department", "nerve specialist"],
    "pediatrics": ["child care", "paediatrics", "peds"],
    "orthopedics": ["bone department", "ortho"],
    "radiology": ["x-ray department", "imaging"],
    "appointment": ["booking", "reservation", "appt"],
    "aiims": ["all india institute of medical sciences"], # Example hospital name
    # Add more synonyms as needed
}

def expand_query_with_synonyms(query): # Logic remains the same, uses updated SYNONYM_MAP
    query_lower = query.lower()
    variants = set([query_lower])
    for keyword, synonyms_list in SYNONYM_MAP.items():
        all_variants_for_keyword = [keyword] + synonyms_list # keyword itself + its synonyms
        # Check if any of the variants (keyword or its synonyms) are in the current query variants
        for current_variant_text in list(variants): # Iterate over a copy as variants set might change
            for term_to_find in all_variants_for_keyword:
                # Use word boundaries for more precise matching
                pattern = re.compile(rf'\b{re.escape(term_to_find)}\b', re.IGNORECASE)
                if pattern.search(current_variant_text):
                    # If found, replace it with all other variants (keyword and its other synonyms)
                    for replacement_term in all_variants_for_keyword:
                        if replacement_term.lower() != term_to_find.lower(): # Don't replace with itself
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
                if score > 0: # Only consider positive scores
                    # If doc index 'i' already scored, take the max score from different query variants
                    all_scored_docs[i] = max(all_scored_docs.get(i, 0.0), score)
        except Exception as e: logger.error(f"Error getting BM25 scores for variant '{q_variant}': {e}"); continue
    
    # Ensure indices are valid for bm25_corpus_docs
    valid_indices = [idx for idx in all_scored_docs.keys() if idx < len(bm25_corpus_docs)]
    # Sort valid indices based on their scores in descending order
    sorted_indices = sorted(valid_indices, key=lambda i: all_scored_docs[i], reverse=True)
    
    top_docs = [bm25_corpus_docs[i] for i in sorted_indices[:k]]
    logger.info(f"BM25 (hospital) retrieved {len(top_docs)} docs for query '{query}'.")
    return top_docs

def detect_and_translate(text, target_lang="en"):
    try:
        detected_lang = detect(text)
        if detected_lang == target_lang: return text, detected_lang
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
    query_lower = normalize_query(query) # Use normalized query
    entities = {
        "hospitals": [], "buildings": [], "floors": [], "rooms": [],
        "departments": [], "doctors": [], "services": [],
        "lifts": [], "stairs": [], "washrooms": [], "general_terms": []
    }

    # Hospital Names (example, make more robust or use known list)
    hospital_matches = re.findall(r'\b(aiims(?:\s+\w+)?|apollo|fortis|max\s+healthcare|city\s+hospital)\b', query_lower, re.IGNORECASE)
    for m in hospital_matches: entities["hospitals"].append(m.strip())
    if not entities["hospitals"] and "hospital" in query_lower : # Generic hospital term
        entities["hospitals"].append("hospital")


    # Building/Block/Wing
    building_matches = re.findall(r'\b(block\s*[\w\d-]+|building\s*[\w\d-]*|tower\s*[\w\d-]*|wing\s*[\w\d-]+|diagnostic\s*block)\b', query_lower, re.IGNORECASE)
    for m in building_matches: entities["buildings"].append(m.strip())

    # Floor Numbers
    floor_matches = re.findall(r'(?:floor|level|flr)\s*(\d+[-\w]*\b)|(\b\d+)(?:st|nd|rd|th)?\s*(?:floor|level|flr)|(ground\s*floor|gf\b)', query_lower, re.IGNORECASE)
    for m_tuple in floor_matches:
        val = next(filter(None, m_tuple), None) # Get the first non-None value from the tuple
        if val:
            if "ground floor" in val or "gf" == val : entities["floors"].append("0")
            else: entities["floors"].append(re.sub(r'[^\d\w-]', '', val)) # Clean non-alphanumeric except dash

    # Room Numbers/Names (generic, may need context)
    room_matches = re.findall(r'\b(?:room|rm|cabin|opd)\s*([\w\d-]+)\b|(\b\d+[A-Za-z]?-?\d*[A-Za-z]?\b(?!\s*floor|\s*st|\s*nd|\s*rd|\s*th|\s*am|\s*pm))', query_lower, re.IGNORECASE)
    for m_tuple in room_matches:
        val = next(filter(None, m_tuple), None)
        if val and len(val) > 1 : entities["rooms"].append(val.strip())

    # Departments (extend with more common department names)
    department_keywords = [
        "cardiology", "neurology", "oncology", "pediatrics", "paediatrics", "radiology", "surgery", "opd", "outpatient",
        "emergency", "casualty", "icu", "intensive care", "orthopedics", "ortho", "gynecology", "gynaecology",
        "dermatology", "ent", "ear nose throat", "urology", "psychiatry", "pathology", "laboratory", "lab",
        "pharmacy", "physiotherapy", "anesthesia", "dental", "opthalmology", "eyes"
    ]
    # Create a regex pattern for departments: \b(keyword1|keyword2|...)\s*(?:department|dept|clinic|unit|ward|center)?\b
    # Or simpler: find keywords and if "department", "clinic" etc. are nearby.
    for dept_kw in department_keywords:
        if re.search(rf'\b{dept_kw}\b', query_lower, re.IGNORECASE):
            entities["departments"].append(dept_kw)
    # Also match "cardiology department", "opd clinic" etc.
    dept_matches = re.findall(r'\b(' + '|'.join(department_keywords) + r')\s*(?:department|dept|clinic|unit|ward|center|section)\b', query_lower, re.IGNORECASE)
    for m in dept_matches: entities["departments"].append(m.strip())


    # Doctors (simple Dr. pattern)
    doctor_matches = re.findall(r'\b(?:dr|doctor)\s*([\w\s\.-]+)\b', query_lower, re.IGNORECASE)
    for m_tuple in doctor_matches: # m_tuple is a list of capturing groups if any, or the full match
         name_part = m_tuple if isinstance(m_tuple, str) else (m_tuple[0] if m_tuple else None)
         if name_part:
            # Attempt to remove general titles that might be caught if not specific like "dr smith"
            name_part_cleaned = re.sub(r'\b(department|clinic|hospital|ward|unit)\b', '', name_part, flags=re.IGNORECASE).strip()
            if len(name_part_cleaned.split()) >= 1 and len(name_part_cleaned) > 3: # Avoid very short or generic terms
                 entities["doctors"].append(name_part_cleaned)


    # Services (example keywords, extend significantly)
    service_keywords = ["x-ray", "mri", "ct scan", "ultrasound", "ecg", "blood test", "consultation", "therapy", "checkup", "vaccination", "dialysis"]
    for svc_kw in service_keywords:
        if re.search(rf'\b{re.escape(svc_kw)}\b', query_lower, re.IGNORECASE): # escape for "ct scan"
            entities["services"].append(svc_kw)

    # Lifts, Stairs, Washrooms (generic)
    if re.search(r'\b(lift|elevator)\b', query_lower): entities["lifts"].append("lift")
    if re.search(r'\b(stairs|staircase)\b', query_lower): entities["stairs"].append("stairs")
    if re.search(r'\b(washroom|toilet|restroom|lavatory|wc)\b', query_lower): entities["washrooms"].append("washroom")

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
                            all_forms = [syn_keyword] + syn_list
                            for form in all_forms:
                                if re.search(rf'\b{re.escape(form)}\b', query_lower, re.IGNORECASE):
                                    entities[entity_type].append(known_item) # Add canonical known_item
                                    break # Found a synonym for this known_item
                            break # Moved to next known_item


    for k in entities: entities[k] = sorted(list(set(e.strip() for e in entities[k] if e and len(e.strip()) > 1))) # Deduplicate and clean
    
    # Remove "dr" or "doctor" if it's the only thing in doctors list and other doctor names are present
    if "dr" in entities["doctors"] and len(entities["doctors"]) > 1: entities["doctors"].remove("dr")
    if "doctor" in entities["doctors"] and len(entities["doctors"]) > 1: entities["doctors"].remove("doctor")


    if any(entities.values()): logger.info(f"[RuleBased NER] Extracted entities from '{query}': {entities}")
    return entities


def detect_task_type_rule_based(query): # Adapted for hospital
    query_l = query.lower()
    if any(kw in query_l for kw in ["list all", "show all doctors", "all departments", "every service"]): return "listing_all"
    if any(kw in query_l for kw in ["list services", "summarize treatments", "overview of doctors"]): return "listing_specific"
    if any(kw in query_l for kw in ["where is", "location of", "find near", "how to reach", "direction to", "room number", "which floor"]): return "location"
    if any(kw in query_l for kw in ["email of", "contact for", "phone number of", "call dr", "website for hospital"]): return "contact_info"
    if any(kw in query_l for kw in ["book appointment", "appointment with", "availability of doctor", "reserve slot", "schedule visit"]): return "booking_info" # For appointments
    if any(kw in query_l for kw in ["how to ", "explain procedure", "what are symptoms", "details about disease", "treatment for"]): return "explanation"
    if any(kw in query_l for kw in ["compare treatments", "difference between doctors", "service A vs service B"]): return "comparison"
    if any(kw in query_l for kw in ["operating hours", "timings", "when is opd open", "doctor schedule", "visiting hours"]): return "operating_hours" # or doctor_availability
    if any(kw in query_l for kw in ["doctor availability", "is dr available", "dr schedule"]): return "doctor_availability"
    if any(kw in query_l for kw in ["department of", "cardiology services", "info on neurology dept"]): return "department_info"
    if any(kw in query_l for kw in ["service offered", "x-ray available", "mri cost info"]): return "service_info"

    # Out of scope - keep general
    if any(kw in query_l for kw in ["weather", "time now", "news", "stock price", "meaning of life", "who are you", "what is your name"]): return "out_of_scope"
    return "general_information" # Default

from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# Ensure the dictionary path is correct or handle potential FileNotFoundError
try:
    sym_spell.load_dictionary("resources/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
except Exception as e:
    logger.warning(f"Could not load spelling dictionary: {e}. Spelling correction might be affected.")


def correct_spelling(text, verbose=False):
    if not sym_spell.words: # Check if dictionary loaded
        logger.warning("SymSpell dictionary not loaded. Skipping spell correction.")
        return text
    try:
        suggestions = sym_spell.lookup_compound(text.lower(), max_edit_distance=2, transfer_casing=True) # transfer_casing=True
        if suggestions:
            corrected = suggestions[0].term
            if corrected.lower() != text.lower(): # Log only if actual correction happened
                 logger.info(f"[SpellCheck] Corrected '{text}' → '{corrected}'")
            return corrected
    except Exception as e:
        logger.error(f"Error during spell correction for '{text}': {e}")
    return text # Return original text if no suggestions or error

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

    # More precise matching for short queries
    if len(query_clean.split()) <= 3:
        # Exact matches first for common short phrases
        if query_clean in greeting_variants: return "greeting"
        if query_clean in exit_variants: return "exit"
        if query_clean in appreciation_variants: return "appreciation"
        if query_clean in confirmation_variants: return "confirmation"
        if query_clean in negation_variants: return "negation"
        # Then fuzzy for very short ones if no exact match
        if any(fuzz.ratio(query_clean, var) >= 90 for var in greeting_variants if len(var) <= len(query_clean) + 2): return "greeting"


    # Fuzzy matching for longer queries or slight variations
    # Thresholds might need tuning based on observed misclassifications
    # Using partial_ratio can be too broad. ratio or token_set_ratio might be better.
    # For robustness, checking token_set_ratio
    match_threshold = 88 # Maintained threshold
    
    # Check for more specific intents first if they are longer phrases
    if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in appreciation_variants): return "appreciation"

    # Then broader categories
    if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in greeting_variants): return "greeting"
    if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in exit_variants): return "exit"
    
    # Smalltalk is tricky, ensure it doesn't override actual queries that might contain "good", "ok"
    # Only classify as smalltalk if it's a strong match and doesn't look like part of a question/command
    if not re.search(r'(what|who|where|when|why|how|list|find|tell me|search|give me|i need|can you)\b', query_clean):
        if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in smalltalk_variants): return "smalltalk"

    if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in confirmation_variants): return "confirmation"
    if any(fuzz.token_set_ratio(query_clean, var) >= match_threshold for var in negation_variants): return "negation"
    
    return None


def is_likely_room_code(token: str) -> bool: # General, can be kept
    return bool(re.match(r"^\d+[a-z]([-_\s]?\d+[a-z])?$", token, re.IGNORECASE)) or \
           bool(re.match(r"^[A-Za-z]?\d{2,}[A-Za-z]?$", token)) # e.g. R101, 303B

def normalize_room_code(token: str) -> str: # General, can be kept
    token = re.sub(r"[-_\s]+", "", token) # Remove separators: 3-a-2b -> 3a2b
    # Standardize: e.g. 3a2b -> 3A2B (if desired, or keep case)
    # For now, keep original casing from token, just remove separators.
    return token.upper() # Standardize to uppercase for consistency

def normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = collapse_repeated_letters(q) # helllo -> helo

    # Standardize medical/hospital suffixes or common terms
    q = q.replace("dept.", "department")
    q = q.replace("dr.", "doctor") # Standardize "dr." to "doctor" for easier matching later
    
    # Normalize room/OPD patterns like "opd3", "room 101a"
    q = re.sub(r"\b(opd|room|rm|cabin|ward|icu)\s*(\d+[a-z]?)", r"\1-\2", q, flags=re.IGNORECASE) # opd 3 -> opd-3
    q = re.sub(r"\b(\d+[a-z]?)\s*(opd|room|rm|cabin|ward|icu)", r"\2-\1", q, flags=re.IGNORECASE) # 3a opd -> opd-3a

    # Tokenize and normalize specific codes if necessary (like room codes)
    # Kept simple for now, relying on downstream NER/regex for structured entities.

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
    # Example: If a specific doctor is mentioned but not department, or vice-versa
    if entities.get("doctors") and not entities.get("departments"):
        # This requires knowing which departments doctors belong to, which is complex here.
        # Simpler: "Which department is Dr. [name] in?" (if LLM can't find)
        suggestions.append(f"Could you specify which department Dr. {entities['doctors'][0]} belongs to?")

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
    query_l = query.lower(); style, tone = "paragraph", "professional_and_helpful"
    if any(word in query_l for word in ["bullet points", "list them"]): style = "bullet_list"
    if any(word in query_l for word in ["in a table", "tabular format"]): style = "table"
    if any(word in query_l for word in ["casual", "friendly", "informal"]): tone = "friendly_and_casual"
    if any(word in query_l for word in ["formal", "official statement"]): tone = "formal_and_precise"
    return style, tone

def rewrite_query_with_memory(query: str, memory: ConversationMemory):
    """Rewrites the user's query to be self-contained using conversation history and summary."""
    original_query = query.strip()
    if not memory.history and not memory.summary: # No context to use for rewriting
        return original_query

    # Don't rewrite very long queries or queries that seem self-contained
    if len(original_query.split()) > 15 or original_query.lower().startswith(("what is", "who is", "explain")):
        logger.info(f"[Coref Rewrite] Skipped for long/self-contained query: '{original_query}'")
        return original_query

    # Get history and summary for the prompt
    context_history = memory.get_contextual_history_text(num_turns=4)

    # Check if there's enough history to warrant a rewrite
    if not context_history.strip() or len(memory.history) < 1:
        return original_query

    rewrite_prompt_template = f"""You are an expert in query understanding. Your task is to rewrite a follow-up user query to be a self-contained question. Use the provided conversation history to resolve any pronouns (like "it", "their", "his"), ambiguities, and follow-up questions (like "what about its location?").

    Conversation Context:
    ---
    {context_history}
    ---

    Follow-up User Query: "{original_query}"

    Instructions:
    1. If the Follow-up User Query is already self-contained and clear, return it as is.
    2. Otherwise, rewrite it into a clear, specific, and self-contained question.
    3. Do NOT answer the question. Only provide the rewritten query.

    Rewritten Query:"""

    try:
        if not memory.llm:
            logger.warning("[Coref Rewrite] LLM not available in memory object. Skipping rewrite.")
            return original_query

        logger.info(f"[Coref Rewrite] Attempting to rewrite query: '{original_query}'")
        response = memory.llm.invoke(rewrite_prompt_template)
        rewritten_query = response.content.strip().strip('"')

        # Post-processing to ensure it's a valid query
        if not rewritten_query or len(rewritten_query) < len(original_query) - 10:
            logger.warning(f"[Coref Rewrite] Rewritten query is shorter or empty. Using original. Rewritten: '{rewritten_query}'")
            return original_query

        if rewritten_query.lower() != original_query.lower():
            logger.info(f"[Coref Rewrite] Rewrote '{original_query}' → '{rewritten_query}'")
        else:
            logger.info(f"[Coref Rewrite] Query '{original_query}' was deemed self-contained.")
        
        return rewritten_query

    except Exception as e:
        logger.error(f"[Coref Rewrite] Error during query rewriting: {e}")
        return original_query # Fallback to original query on error


# Small talk handler, largely the same, LLM prompt context updated if needed
def handle_small_talk(user_query, memory, session):
    import random
    convo_prompt = f"""
You are a cheerful, helpful assistant at the Aiims building in Aiims Jammu. The user is being casual, friendly, or saying goodbye. Reply in a varied, natural, warm tone — use emojis sometimes. Avoid repeating yourself.

User: {user_query}
Assistant:"""

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.85)
        response = llm.invoke(convo_prompt).content
    except Exception as e:
        logger.error(f"Error in small talk handler: {e}")
        response = random.choice([
            "Hey there! 👋 What can I help you with at RNI?",
            "Hi! 😊 Feel free to ask about rooms or companies.",
            "I'm here! What info are you looking for today?"
        ])

    if not hasattr(memory, "history") or not isinstance(memory.history, list):
        memory.history = []

    memory.history.append({
        "turn_index": len(memory.history),
        "user": user_query,
        "assistant": response
    })

    session["memory"] = memory.__dict__
    return {"answer": response}


def chat(user_query: str):
    request_start_time = datetime.now()
    logger.info(f"--- New Chat Request (Hospital) --- Query: '{user_query}'")
    if "memory" not in session: 
        # Initialize with the LLM instance for summarization capabilities
        session["memory"] = ConversationMemory(llm=ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.2)).__dict__
    
    conv_memory = ConversationMemory(llm=ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.2)) # Fresh instance for current request logic
    try:
        # Load only known attributes of ConversationMemory from session
        valid_keys = conv_memory.__dict__.keys()
        session_memory_dict = session.get("memory", {}) # Use .get for safety
        loadable_memory = {k: session_memory_dict[k] for k in valid_keys if k in session_memory_dict}
        conv_memory.__dict__.update(loadable_memory)
    except Exception as e:
        logger.error(f"Error loading conversation memory from session: {e}. Initializing new memory.")
        session["memory"] = ConversationMemory().__dict__ # Reset session memory on error


    original_user_query = user_query.strip()
    query_lower_raw = original_user_query.lower()

    # Early check for conversational intents that might not need full processing pipeline
    # Hospital entity keywords to prevent premature small talk classification for queries like "hello, where is cardiology?"
    hospital_entity_keywords = [
        "room", "opd", "icu", "ward", "doctor", "dr", "nurse", "staff", "physician", "consultant", "specialist",
        "department", "cardiology", "neurology", "oncology", "pediatrics", "radiology", "surgery", "clinic",
        "service", "x-ray", "mri", "scan", "test", "appointment", "treatment", "procedure",
        "hospital", "aiims", "building", "floor", "location", "find", "where", "contact", "phone", "email",
        "availability", "schedule", "hours", "timings"
    ]
    
    # Detect conversational intent BEFORE translation or complex normalization
    convo_intent = detect_conversational_intent(original_user_query) 
    
    # If conversational intent detected AND query doesn't seem to contain substantial entity keywords, handle as small talk
    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation"}:
        if not any(keyword in query_lower_raw for keyword in hospital_entity_keywords) or len(original_user_query.split()) <= 3 :
             logger.info(f"Handling as small talk due to intent '{convo_intent}' and lack of strong entity keywords / short query.")
             # Pass original query to small talk handler to preserve nuance
             small_talk_response = handle_small_talk(original_user_query, conv_memory, session) # Ensure handle_small_talk saves memory
             return small_talk_response


    # Proceed with full pipeline for substantive queries or mixed conversational/substantive queries
    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
    translated_query, detected_input_lang = detect_and_translate(cleaned_query_for_lang_detect)
    
    # Query normalization and coreference resolution
    normalized_translated_query = normalize_query(translated_query) # Normalize after translation
    processed_query = rewrite_query_with_memory(normalized_translated_query, conv_memory)
    
    logger.info(f"Original Query: '{original_user_query}', Translated: '{translated_query}', Normalized: '{normalized_translated_query}', Processed (Coref): '{processed_query}'")

    # NLU processing on the potentially rewritten and normalized query
    extracted_query_entities = nlu_processor.extract_entities(processed_query)
    task_type = nlu_processor.classify_intent(processed_query) # Classify intent on processed query
    
    # If a greeting was part of a substantive query, prepare greeting response
    # This needs to be handled carefully to avoid double greeting if already handled by small_talk block
    greeting_resp_prefix = ""
    if convo_intent == "greeting" and any(keyword in query_lower_raw for keyword in hospital_entity_keywords):
        # Generate a simple greeting to prepend, but don't save to memory here as main response will be saved.
        # Or let the LLM handle the greeting naturally if it's part of the prompt.
        # For simplicity, let LLM handle natural greeting if query is mixed.
        logger.info("Query seems to be a mixed greeting + substantive question. LLM will handle.")


    # Add current turn to conversation memory (user query, placeholder for assistant, extracted entities)
    # Assistant response will be filled later.
    conv_memory.add_turn(original_user_query, "", extracted_query_entities) # Store original user query
    logger.info(f"Detected task type (hospital): {task_type}, Entities: {extracted_query_entities}")


    if task_type == "out_of_scope":
        # Use hospital-specific OOS message
        out_of_scope_response = "I am an assistant for this hospital and can only answer questions related to its facilities, departments, doctors, services, and appointments. How can I help you with that?"
        conv_memory.history[-1]["assistant"] = out_of_scope_response # Update last turn's assistant response
        session["memory"] = conv_memory.__dict__
        if target_lang_code and target_lang_code != "en": out_of_scope_response = GoogleTranslator(source="en", target=target_lang_code).translate(out_of_scope_response)
        elif detected_input_lang != "en" and detected_input_lang is not None : out_of_scope_response = GoogleTranslator(source="en", target=detected_input_lang).translate(out_of_scope_response)
        return {"answer": out_of_scope_response, "debug_info": {"task_type": task_type, "original_query": original_user_query, "processed_query": processed_query}}

    if not GROQ_API_KEY: logger.critical("Groq API key not configured."); return {"answer": "Error: Chat service temporarily unavailable."}
    
    query_chars = classify_query_characteristics(processed_query)
    response_length_hint = query_chars.get("response_length", "medium") # Default to medium
    answer_style, answer_tone = detect_answer_style_and_tone(processed_query)
    logger.info(f"Response hints (hospital): length={response_length_hint}, style={answer_style}, tone={answer_tone}")

    retrieved_docs = hybrid_retriever(processed_query, k_simple=6, k_normal=10, k_complex=15) # Increased k for potentially more complex data
    
    if not retrieved_docs:
        logger.warning(f"No documents retrieved for hospital query: {processed_query}")
        clarification_msg = "I couldn't find specific information for your query. "
        suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
        clarification_msg += " ".join(suggestions) if suggestions else "Could you try rephrasing or provide more details?"
        
        conv_memory.history[-1]["assistant"] = clarification_msg
        session["memory"] = conv_memory.__dict__
        
        if target_lang_code and target_lang_code != "en": clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en" and detected_input_lang is not None: clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)
        return {"answer": clarification_msg, "related_queries": suggestions if suggestions else [], "debug_info": {"task_type": task_type, "original_query": original_user_query, "processed_query": processed_query, "entities": extracted_query_entities}}

    bi_reranked_docs = rerank_documents_bi_encoder(processed_query, retrieved_docs, top_k=8) # Increase bi-encoder top_k
    final_docs_for_llm = rerank_documents_cross_encoder(processed_query, bi_reranked_docs, top_k=4) # Increase cross-encoder top_k

    # Entity grounding check
    entity_terms_to_check = set()
    critical_entity_types = ["doctors", "departments", "services", "rooms", "hospitals"] # Focus on key entities
    for ent_type in critical_entity_types:
        for val_list in ensure_list(extracted_query_entities.get(ent_type, [])):
            for val in ensure_list(val_list): # Handle if val_list contains sub-lists (should not based on current NER output)
                 val_clean = str(val).lower().strip() # Ensure val is string
                 if val_clean and len(val_clean) > 2 and not val_clean.startswith("##"): # Min length 3
                    entity_terms_to_check.add(val_clean)
                    # Add parts of multi-word entities
                    if ' ' in val_clean: entity_terms_to_check.update(val_clean.split())


    logger.info(f"[Entity Grounding] Checking for terms in docs: {entity_terms_to_check}")
    missing_entities_in_docs = []
    if entity_terms_to_check and final_docs_for_llm: # Only check if there are terms and docs
        for term in entity_terms_to_check:
            if term and len(term) > 2: # Avoid checking very short/generic terms like "dr" alone
                found_in_docs = any(term in doc.page_content.lower() for doc in final_docs_for_llm)
                if not found_in_docs:
                    missing_entities_in_docs.append(term)
    
    # If critical entities are extracted from query but not found in top retrieved docs, consider it a low-confidence retrieval
    # This is a heuristic. The LLM might still find it if the rerankers missed something obvious.
    # Or, the entity was hallucinated by NER.
    if missing_entities_in_docs and task_type not in ["general_information", "listing_all"] and len(entity_terms_to_check) > 0 : # Be more stringent for specific tasks
        # Check if at least one primary entity type is completely missing
        primary_types_mentioned = [et for et in critical_entity_types if extracted_query_entities.get(et)]
        all_primary_entities_missing_from_docs = True
        if primary_types_mentioned:
            for p_type in primary_types_mentioned:
                type_entities = [str(v).lower().strip() for v_list in ensure_list(extracted_query_entities.get(p_type,[])) for v in ensure_list(v_list)]
                if any(any(str(t_e).lower().strip() in doc.page_content.lower() for t_e in type_entities if len(str(t_e).strip())>2) for doc in final_docs_for_llm):
                    all_primary_entities_missing_from_docs = False
                    break
        
        if all_primary_entities_missing_from_docs and primary_types_mentioned:
            logger.warning(f"[Entity Grounding - Critical Miss] Key extracted entities ({missing_entities_in_docs}) not found in top retrieved documents. This may lead to poor answer quality.")
            # Decide if to send to LLM or ask for clarification. For now, proceed to LLM but log warning.
            # Could add a flag to debug_info or even ask for clarification here.
            # Example clarification:
            # clarification_msg = f"I found some general information, but couldn't pinpoint details for: {', '.join(missing_entities_in_docs)}. Could you be more specific or try different terms?"
            # ... (translate and return) ...
            # For now, let LLM attempt to answer, it might still manage with broader context.


    # Force-inject top BM25 document if different and seems relevant
    top_bm25_docs_for_injection = bm25_retriever_func(processed_query, k=1) # Get just the top 1
    if top_bm25_docs_for_injection:
        top_bm25_doc = top_bm25_docs_for_injection[0]
        # Check if its content is not already in final_docs_for_llm
        is_top_bm25_already_present = any(
            top_bm25_doc.page_content.strip() == doc.page_content.strip() for doc in final_docs_for_llm
        )
        if not is_top_bm25_already_present:
            # Heuristic: Add if it contains some of the extracted entities not well covered by reranked docs
            # For simplicity now, just add it if it's different.
            final_docs_for_llm.append(top_bm25_doc) # Append, LLM can choose
            logger.info("Injected top BM25 doc into LLM context as it was different.")
            # Re-limit if too many docs, e.g., final_docs_for_llm = final_docs_for_llm[:5]

    logger.info(f"Final {len(final_docs_for_llm)} documents selected for LLM context (hospital).")
    if not final_docs_for_llm and retrieved_docs: # Fallback if reranking somehow lost all docs
        final_docs_for_llm = retrieved_docs[:3] # Take top from initial retrieval
        logger.warning("Reranking resulted in zero documents. Using top 3 from initial hybrid retrieval for LLM.")
    elif not final_docs_for_llm and not retrieved_docs: # No docs at all
        logger.error("No documents available at all to send to LLM.")
        # This case should ideally be caught earlier (after hybrid_retriever)
        # but as a final safeguard:
        no_docs_msg = "I'm sorry, I couldn't find any information related to your query at the moment."
        conv_memory.history[-1]["assistant"] = no_docs_msg
        session["memory"] = conv_memory.__dict__
        if target_lang_code and target_lang_code != "en": no_docs_msg = GoogleTranslator(source="en", target=target_lang_code).translate(no_docs_msg)
        elif detected_input_lang != "en" and detected_input_lang is not None: no_docs_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(no_docs_msg)
        return {"answer": no_docs_msg, "debug_info": {"error": "No documents for LLM", "task_type": task_type, "processed_query": processed_query}}


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
        filtered_meta_info = {k: v for k, v in meta_info.items() if v and str(v).strip() and str(v).lower() != "n/a"} # Filter out empty/None/NA
        if filtered_meta_info:
            doc_text += "Key Metadata: " + "; ".join([f"{k}: {v}" for k, v in filtered_meta_info.items()])
        context_parts.append(doc_text)
    extracted_context_str = "\n\n---\n\n".join(context_parts)
    
    hospital_name_for_prompt = "Aiims Jammu"
    if data_loader.building_data and isinstance(data_loader.building_data, list) and len(data_loader.building_data) > 0:
        hospital_name_for_prompt = data_loader.building_data[0].get("hospitalName", "Aiims Jammu")


    prompt_intro = f"You are an advanced, intelligent, and conversational AI assistant for '{hospital_name_for_prompt}'. Your primary goal is to provide accurate, concise, and relevant information based ONLY on the 'Extracted Context' provided. If the context is insufficient or irrelevant, clearly state that you cannot answer or need more information. Do NOT invent information or use external knowledge."
    
    task_instructions = "" # Default empty
    if task_type in ["location", "location_specific", "location_general"]:
        task_instructions = "Focus on providing location details: hospital name, building, floor, room name/number, and zone if available in context. Mention nearby landmarks or access points if relevant from context."
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
1. Base answers ONLY on 'Extracted Context'. If the information is not in the context, state that clearly (e.g., "Based on the provided information, I cannot answer that," or "The context does not contain details about X."). Do NOT use any external knowledge or make assumptions.
2. If the Extracted Context is empty or clearly irrelevant to the query, state that you lack the necessary information to answer.
3. Consider 'Past Conversation History' for resolving ambiguities (like "his email" referring to a previously discussed doctor) but prioritize the current query and the 'Extracted Context' as the source of truth for the answer.
4. If the query is ambiguous despite context and history, you can ask ONE brief clarifying question.
5. Be conversational, empathetic, and helpful, adapting to a hospital setting.
6. {task_instructions}
7. If asked about medical advice, conditions, or treatments, state that you are an AI assistant and cannot provide medical advice. Suggest consulting with a healthcare professional. However, if the query is about *information available in the context* regarding a service or procedure (e.g., "what does the context say about X-ray procedure?"), then answer based on the context.

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
    chat_history_for_prompt = conv_memory.get_contextual_history_text(num_turns=3) # Slightly less history for prompt
    llm_input_data = {"input": processed_query, "context": extracted_context_str, "history": chat_history_for_prompt}

    # LLM model selection based on complexity/task for hospital queries
    if response_length_hint == "long" or task_type in ["explanation", "comparison", "listing_all"] or "complex" in detect_query_complexity(processed_query) :
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.3 # Slightly lower temp for factual long
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
        final_response_text = ai_message.content.strip() # Strip whitespace
        logger.info(f"LLM Raw Response Snippet (hospital): {final_response_text[:250]}...")
    except Exception as e:
        logger.error(f"Error invoking RAG chain with Groq (hospital): {e}")
        # More specific error if context might be too large (common with Groq/LLMs)
        if "context length" in str(e).lower() or "token limit" in str(e).lower():
            final_response_text = "I apologize, but I encountered an issue processing the information, possibly due to its size. Could you try a more specific question?"
        else:
            final_response_text = "I apologize, but I encountered an issue while processing your request for hospital information. Please try again."

    # Update memory with the final assistant response
    if conv_memory.history: # Ensure history is not empty
        conv_memory.history[-1]["assistant"] = final_response_text
    else: # Should not happen if add_turn was called, but as a safeguard
        conv_memory.add_turn(original_user_query, final_response_text, extracted_query_entities)

    # If there was a greeting in a mixed query, prepend it (handled by LLM now)
    # if greeting_resp_prefix:
    #    final_response_text = f"{greeting_resp_prefix}\n\n{final_response_text}"

    session["memory"] = conv_memory.__dict__ # Save updated memory to session

    # Translate response if needed
    try:
        if target_lang_code and target_lang_code != "en": 
            final_response_text = GoogleTranslator(source="en", target=target_lang_code).translate(final_response_text)
            logger.info(f"Translated response to {target_lang_code}.")
        elif detected_input_lang != "en" and detected_input_lang is not None and detected_input_lang != "en": # check detected_input_lang is not None
            final_response_text = GoogleTranslator(source="en", target=detected_input_lang).translate(final_response_text)
            logger.info(f"Translated response back to input language {detected_input_lang}.")
    except Exception as e: logger.warning(f"Failed to translate final response: {e}")
    
    processing_time = (datetime.now() - request_start_time).total_seconds()
    logger.info(f"--- Chat Request Completed (Hospital) --- Time: {processing_time:.2f}s")
    
    debug_info = {
        "original_query": original_user_query,
        "processed_query": processed_query,
        "detected_task_type": task_type,
        "extracted_entities": extracted_query_entities,
        "detected_input_lang": detected_input_lang,
        "target_response_lang": target_lang_code,
        "response_length_hint": response_length_hint,
        "llm_model_used": groq_llm_model_name,
        "retrieved_docs_count_initial": len(retrieved_docs) if retrieved_docs else 0,
        "retrieved_docs_count_final_llm": len(final_docs_for_llm) if final_docs_for_llm else 0,
        "final_doc_ids_for_llm": [doc.metadata.get("source_doc_id","Unknown") for doc in final_docs_for_llm] if final_docs_for_llm else [],
        "missing_entities_in_docs_warning": missing_entities_in_docs if 'missing_entities_in_docs' in locals() and missing_entities_in_docs else [],
        "processing_time_seconds": round(processing_time, 2)
    }
    return {"answer": final_response_text, "debug_info": debug_info}

# Flask routes (/ and /refresh_data, /chat) remain structurally the same
# but will now use the hospital-adapted logic.
@app.route("/")
def home():
    # If you have a different HTML for hospital, change template name
    return render_template("index.html") 

@app.route("/refresh_data", methods=["POST"])
def refresh_data_endpoint():
    logger.info("Hospital data refresh request received.")
    # data_loader is re-initialized within refresh_faiss_and_bm25
    refresh_faiss_and_bm25() # This now uses hospital data paths and logic
    logger.info("Hospital data and retrieval models refreshed successfully.")
    return jsonify({"message": "Hospital data and retrieval models refreshed successfully."}), 200

@app.route("/chat", methods=["GET", "POST"])
def chat_endpoint():
    if request.method == "GET":
        user_message = request.args.get("message", "").strip()
    else: # POST
        data = request.get_json()
        user_message = data.get("message", "").strip() if data else ""

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = chat(user_message) # Calls the main chat logic adapted for hospital
    return jsonify(response)

if __name__ == "__main__":
    # Consider using a different port if the RNI version might also run
    app.run(host='0.0.0.0', port=5001, debug=True) # e.g., port 5001 for hospital version