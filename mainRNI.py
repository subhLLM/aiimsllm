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
                "location", "contact_info", "booking_info", "explanation", "comparison",
                "operating_hours", "listing_all", "listing_specific", "out_of_scope", "general_information"
            ]
            # NER using dslim/bert-base-NER
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
            logger.info("NLU models initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize NLU models: {e}")
            self.intent_classifier = None
            self.ner_pipeline = None

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
        if not self.ner_pipeline:
            logger.warning("NER pipeline not available. Falling back to rule-based extraction.")
            return extract_entities_rule_based(query)
        try:
            entities = {"floors": [], "rooms": [], "companies": [], "lifts": [], "stairs": [], "washrooms": [], "buildings": [], "general_terms": []}
            ner_results = self.ner_pipeline(query)
            for entity in ner_results:
                entity_type = entity["entity_group"].lower()
                entity_value = entity["word"].strip()
                if entity_type == "loc":
                    if "floor" in entity_value.lower() or re.match(r'\d+', entity_value):
                        entities["floors"].append(entity_value)
                    elif "building" in entity_value.lower() or "tower" in entity_value.lower():
                        entities["buildings"].append(entity_value)
                    elif "room" in entity_value.lower() or re.match(r'\w+\d+', entity_value):
                        entities["rooms"].append(entity_value)
                    elif "lift" in entity_value.lower() or "elevator" in entity_value.lower():
                        entities["lifts"].append(entity_value)
                    elif "stairs" in entity_value.lower() or "staircase" in entity_value.lower():
                        entities["stairs"].append(entity_value)
                    elif "washroom" in entity_value.lower() or "restroom" in entity_value.lower():
                        entities["washrooms"].append(entity_value)
                elif entity_type == "org":
                    entities["companies"].append(entity_value)
                elif entity_type == "misc":
                    entities["general_terms"].append(entity_value)
            for key in entities:
                entities[key] = sorted(list(set(entities[key])))
            logger.info(f"Extracted entities from '{query}': {entities}")
            return entities
        except Exception as e:
            logger.error(f"Error in NER: {e}")
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
                        # Track latest entity of each type
                        self.last_entity_by_type[entity_type] = entity_value
            if self.contextual_entities:
                self.current_topic = self.contextual_entities[-1]

        # Prune history if it exceeds limits
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def get_last_entity_by_priority(self, type_priority=["companies", "rooms", "floors", "washrooms", "buildings", "lifts", "stairs"]):
        for entity_type in type_priority:
            if entity_type in self.last_entity_by_type:
                return self.last_entity_by_type[entity_type]
        return None

    def get_contextual_history_text(self, num_turns=5):
        history_text = ""
        
        recent_turns = self.history[-num_turns:] if isinstance(self.history, list) else []

        for i, turn in enumerate(recent_turns):
            turn_index = turn.get("turn_index", i)
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey_for_rni_app_v2")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session/"
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

def ensure_list(val):
    if isinstance(val, list): return val
    elif val: return [val]
    return []

FAISS_INDEX_PATH = "rag_rni_building"
RNI_BUILDING_JSON_PATH = "RNI_building.json"
ALLOWED_FILES = {os.path.basename(RNI_BUILDING_JSON_PATH)}
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY not found in environment variables.")

class RNIDataLoader:
    def __init__(self, filepath=RNI_BUILDING_JSON_PATH):
        self.filepath = filepath
        self.building_data = self.load_json_secure(self.filepath)
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
        if not self.building_data:
            return {
                "rooms": [], "companies": [], "lifts": [], "stairs": [],
                "emergencyExits": [], "entrances": [], "ramps": []
            }

        rooms, companies = [], []
        lifts, stairs, exits, entrances, ramps = set(), set(), set(), set(), set()

        for item in self.building_data:
            physical = item.get("physical", {})
            if physical.get("name"): rooms.append(physical["name"])

            entity = item.get("functional", {}).get("associatedEntity", {})
            if entity.get("name"): companies.append(entity["name"])

            access = item.get("accessibility", {}).get("nearestAccessPoints", {})
            for group, collector in [("lifts", lifts), ("stairs", stairs), ("emergencyExits", exits),
                                    ("entrances", entrances), ("ramps", ramps)]:
                for ap in access.get(group, []):
                    if ap.get("name"): collector.add(ap["name"])

        return {
            "rooms": list(set(rooms)),
            "companies": list(set(companies)),
            "lifts": list(lifts),
            "stairs": list(stairs),
            "emergencyExits": list(exits),
            "entrances": list(entrances),
            "ramps": list(ramps)
        }
    
    def get_all_metadata_tags(self):
        tags = set()
        if not self.building_data:
            return []
        for item in self.building_data:
            meta_tags = item.get("metadata", {}).get("tags", [])
            tags.update(meta_tags)
        return sorted(tags)

    def get_metadata_tag_counts(self):
        counter = Counter()
        if not self.building_data:
            return {}
        for item in self.building_data:
            tags = item.get("metadata", {}).get("tags", [])
            counter.update(tags)
        return dict(counter.most_common())


data_loader = RNIDataLoader()

def format_operating_hours(hours_data):
    if not hours_data: return "N/A"

    order = ["mondayToFriday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
    parts = []
    for day in order:
        if day not in hours_data: continue
        label = day.replace("mondayToFriday", "Mon–Fri").capitalize()
        time = hours_data[day]
        if isinstance(time, dict):
            parts.append(f"{label}: {time.get('start', 'N/A')}–{time.get('end', 'N/A')}")
        else:
            parts.append(f"{label}: {str(time)}")
    return "; ".join(parts)

def format_response_channels(channels_list):
    if not channels_list:
        return "N/A"

    parts = []
    for ch in channels_list:
        ch_type = ch.get('type', 'Unknown')
        description = ch.get('description', '')
        contact = ch.get('contact', {})
        
        # Prefer both phone and email if available
        contact_parts = []
        if contact.get('phone'):
            contact_parts.append(f"Phone: {contact['phone']}")
        if contact.get('email'):
            contact_parts.append(f"Email: {contact['email']}")

        contact_str = ", ".join(contact_parts) or "N/A"
        channel_info = f"{ch_type}"
        if description:
            channel_info += f" ({description})"

        channel_info += f": {contact_str}"

        # Add operating hours if available
        op_hours = format_operating_hours(ch.get('operatingHours'))
        if op_hours != "N/A":
            channel_info += f" [Hours: {op_hours}]"

        parts.append(channel_info)

    return ". ".join(parts)


def prepare_documents():
    if not data_loader.building_data:
        logger.error("Building data not loaded. Cannot prepare documents.")
        return []

    documents = []

    for item_index, item_data in enumerate(data_loader.building_data):
        content_parts = []
        metadata_payload = {
            "source_doc_id": item_data.get("id", f"item_{item_index}"),
            "type": item_data.get("physical", {}).get("type", "UnknownType").lower()
        }

        # Location Context
        loc = item_data.get("locationContext", {})
        content_parts.append(
            f"Location: Bldg '{loc.get('buildingName', 'N/A')}', Tower '{loc.get('tower', 'N/A')}', "
            f"Flr {loc.get('floor', 'N/A')}, Zone '{loc.get('zone', 'N/A')}', {loc.get('campusName', 'N/A')}."
        )
        metadata_payload.update({
            "buildingName": loc.get("buildingName"),
            "tower": loc.get("tower"),
            "floor": str(loc.get("floor", "")),
            "zone": loc.get("zone"),
            "campusName": loc.get("campusName")
        })

        # Physical
        physical = item_data.get("physical", {})
        name = physical.get("name", f"Unnamed {metadata_payload['type']}")
        content_parts.append(
            f"Name: {name} (Type: {physical.get('type', 'N/A')}, SubType: {physical.get('subType', 'N/A')})."
        )
        metadata_payload.update({
            "item_name": name,
            "item_subtype": physical.get("subType")
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
                f"Desc: {entity.get('description', 'N/A')[:100]}... "
                f"Ind: {entity.get('industry', 'N/A')}."
            )
            contact = entity.get("contact", {})
            contact_str = (
                f"Email: {contact.get('email', 'N/A')}, Phone: {contact.get('phone', 'N/A')}, "
                f"Web: {contact.get('website', 'N/A')}, Linkedin: {contact.get('linkedin', 'N/A')}."
            )
            content_parts.append(f"Entity Contact: {contact_str}")
            metadata_payload.update({
                "company_name": entity.get("name"),
                "company_contact_email": contact.get("email"),
                "company_contact_phone": contact.get("phone"),
                "company_website": contact.get("website"),
                "company_linkedin_website": contact.get("linkedin")
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

        # media.images
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

    logger.info(f"Prepared {len(documents)} documents for FAISS index.")
    return documents

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
    model_id = effective_embedding_model.model_name

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

    logger.info("Building new FAISS index.")
    docs_for_faiss = prepare_documents()
    if not docs_for_faiss:
        logger.error("No documents prepared. FAISS index cannot be built.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
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
    documents = prepare_documents()
    if not documents:
        logger.error("No documents available for BM25 initialization.")
        return None

    bm25_corpus_docs = documents
    bm25_tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

    if not bm25_tokenized_corpus:
        logger.error("BM25 tokenized corpus is empty.")
        return None

    logger.info(f"Initialized BM25 with {len(bm25_tokenized_corpus)} documents.")
    return BM25Okapi(bm25_tokenized_corpus)
bm25 = initialize_bm25()

# Initialize NLU Processor
nlu_processor = NLUProcessor()

def get_embedding_model_for_query(query):
    query_lower = query.lower()

    # Hybrid model: Broad/general search or fuzzy context
    if any(word in query_lower for word in ["search", "find", "nearby", "available", "locate", "which room", "what rooms", "where can i find"]):
        return embedding_models["hybrid"]

    # QA-optimized: Direct factual questions
    if any(word in query_lower for word in ["what is", "who is", "where is", "define", "tell me about", "give me info about"]):
        return embedding_models["qa"]

    # Factual: Explanation or in-depth info
    if any(word in query_lower for word in ["explain", "describe", "details about", "how does", "everything about"]):
        return embedding_models["factual"]

    # Ranking: Comparisons or list-based queries
    if any(word in query_lower for word in ["list all", "compare", "rank", "top", "best", "vs", "versus"]):
        return embedding_models["ranking"]

    # Multilingual: Default fallback, good multilingual support
    else:
        model = embedding_models["multilingual"]

    logger.info(f"[Embedding Model Routing] Using: {model.model_name} for query: {query}")
    return model


def refresh_faiss_and_bm25():
    global db_faiss, bm25, faiss_index_cache
    logger.info("Refreshing FAISS index and BM25 model.")

    faiss_index_cache.clear()
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    db_faiss = initialize_faiss()
    bm25 = initialize_bm25()

    if db_faiss and bm25:
        logger.info("FAISS and BM25 refreshed successfully.")
    else:
        logger.error("Refresh failed.")
    return db_faiss, bm25

def detect_query_complexity(query):
    query_lower = query.lower()
    if any(conj in query_lower for conj in [" and ", " or ", " but also ", " as well as "]) and len(query.split()) > 7: return "complex"
    if any(word in query_lower for word in ["list all", "all rooms", "all companies", "explain in detail", "everything about", "how to book", "compare"]): return "complex"
    if len(query.split()) <= 5 and any(q_word in query_lower for q_word in ["where is", "email of", "phone of"]): return "simple"
    return "normal"

def hybrid_retriever(query, k_simple=5, k_normal=8, k_complex=12):
    selected_embedding_instance = get_embedding_model_for_query(query)

    current_db_faiss = (
        faiss_index_cache.get(selected_embedding_instance.model_name)
        or initialize_faiss(selected_embedding_instance)
    )

    if not current_db_faiss:
        logger.error("FAISS database not available.")
        return []

    if not bm25:
        logger.error("BM25 not available. Falling back to FAISS only.")
        return current_db_faiss.as_retriever(search_kwargs={"k": k_normal}).get_relevant_documents(query)

    complexity = detect_query_complexity(query)
    k_val = k_simple if complexity == "simple" else (k_normal if complexity == "normal" else k_complex)

    logger.info(f"Hybrid retrieval for '{query}' → complexity: {complexity}, k={k_val}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(
            current_db_faiss.as_retriever(search_kwargs={"k": k_val}).get_relevant_documents, query
        )
        bm25_future = executor.submit(bm25_retriever_func, query, k_val)

        try:
            faiss_docs = faiss_future.result(timeout=10)
            bm25_top_docs = bm25_future.result(timeout=10)
        except TimeoutError:
            logger.warning("Retrieval timed out.")
            faiss_docs = faiss_future.result() if faiss_future.done() else []
            bm25_top_docs = bm25_future.result() if bm25_future.done() else []
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            faiss_docs, bm25_top_docs = [], []

    all_docs_dict = {doc.page_content: doc for doc in faiss_docs}
    for doc in bm25_top_docs:
        if doc.page_content not in all_docs_dict:
            all_docs_dict[doc.page_content] = doc

    merged_docs = list(all_docs_dict.values())
    logger.info(f"Hybrid merged {len(merged_docs)} docs.")
    return merged_docs[:k_val * 2]

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

SYNONYM_MAP = {"elevator": ["lift"], "lift": ["elevator"], "toilet": ["washroom", "restroom"], "washroom": ["toilet", "restroom"], "restroom": ["toilet", "washroom"], "stairs": ["staircase"], "staircase": ["stairs"], "contactno": ["contact number", "phone no", "phone number"], "iwayplus": ["iwayplus pvt. ltd.", "iway plus", "iwayplus.com"], "timings": ["operating hours", "open hours"], "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0", "gf"], "neo risers": ["neorisers"]}
def expand_query_with_synonyms(query):
    query_lower = query.lower()
    variants = set([query_lower])

    # Apply global substitution for any matching key or synonym
    for keyword, synonyms_list in SYNONYM_MAP.items():
        all_variants = [keyword] + synonyms_list
        for variant in list(variants):  # loop over current variants
            for alt in all_variants:
                pattern = re.compile(rf'\b{re.escape(alt)}\b', re.IGNORECASE)
                if pattern.search(variant):
                    for replacement in all_variants:
                        if replacement.lower() != alt.lower():
                            new_variant = pattern.sub(replacement, variant)
                            variants.add(new_variant)

    final_variants = list(set(v for v in variants if len(v) > 3))[:6]  # limit to 6 variants
    if len(final_variants) > 1:
        logger.info(f"Expanded query '{query}' to variants: {final_variants}")
    return final_variants

def bm25_retriever_func(query, k=10):
    if not bm25 or not bm25_corpus_docs: logger.warning("BM25 model or corpus not initialized."); return []
    expanded_queries = expand_query_with_synonyms(query); all_scored_docs = {}
    for q_variant in expanded_queries:
        tokenized_query = q_variant.lower().split()
        if not tokenized_query: continue
        try:
            scores = bm25.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                if score > 0: all_scored_docs[i] = max(all_scored_docs.get(i, 0.0), score)
        except Exception as e: logger.error(f"Error getting BM25 scores for variant '{q_variant}': {e}"); continue
    valid_indices = [idx for idx in all_scored_docs.keys() if idx < len(bm25_corpus_docs)]
    sorted_indices = sorted(valid_indices, key=lambda i: all_scored_docs[i], reverse=True)
    top_docs = [bm25_corpus_docs[i] for i in sorted_indices[:k]]
    logger.info(f"BM25 retrieved {len(top_docs)} docs for query '{query}'.")
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
    query_lower = normalize_query(query)
    entities = {"floors": [], "rooms": [], "companies": [], "lifts": [], "stairs": [], "washrooms": [], "buildings": [], "general_terms": []}
    floor_matches = re.findall(r'(?:floor|level|flr)\s*(\d+[-\w]*\b)|(\b\d+)(?:st|nd|rd|th)?\s*(?:floor|level|flr)|(ground\s*floor|gf\b)', query_lower)
    for m in floor_matches:
        val = next(filter(None, m), None)
        if val == "ground floor" or val == "gf": entities["floors"].append("0")
        elif val: entities["floors"].append(re.sub(r'[^\d\w-]', '', val))
    room_matches = re.findall(r'\b(?:room|office|cabin|space|lab|hall|mr)\s*([\w\d-]+)\b|(\b\d+[A-Za-z]?-?\d*[A-Za-z]?\b(?!\s*floor))', query_lower)
    for m in room_matches:
        val = next(filter(None, m), None)
        if val and len(val) > 1: entities["rooms"].append(val.strip())
    lift_matches = re.findall(r'\b(lift\s*lobby\s*[\w\d-]+|lift\s*[\w\d-]*|elevator\s*[\w\d-]*)\b', query_lower)
    for m in lift_matches: entities["lifts"].append(m.strip())
    if re.search(r'\b(stairs|staircase|stairwell)\b', query_lower): entities["stairs"].append("stairs")
    if re.search(r'\b(washroom|toilet|restroom|lavatory|wc)\b', query_lower): entities["washrooms"].append("washroom")
    building_matches = re.findall(r'\b(block\s*[\w\d-]+|building\s*[\w\d-]*|tower\s*[\w\d-]*)\b', query_lower)
    for m in building_matches: entities["buildings"].append(m.strip())

    # Boost: Look for known company names directly in query (even if NER fails)
    known_companies = data_loader.all_known_entities.get("companies", [])
    for company in known_companies:
        canonical = company.lower()
        if canonical in query_lower:
            entities["companies"].append(company)
        else:
            for alias in SYNONYM_MAP.get(canonical, []):
                if alias.lower() in query_lower:
                    entities["companies"].append(company)
                    break

    for k in entities: entities[k] = sorted(list(set(entities[k])))
    if any(entities.values()): logger.info(f"Extracted entities from '{query}': {entities}")
    return entities

def detect_task_type_rule_based(query):
    query_l = query.lower()
    if any(kw in query_l for kw in ["list all", "show all rooms", "give all companies"]): return "listing_all"
    if any(kw in query_l for kw in ["list", "summarize rooms", "overview of companies"]): return "listing_specific"
    if any(kw in query_l for kw in ["where is", "location of", "find near", "how to reach", "direction to"]): return "location"
    if any(kw in query_l for kw in ["email of", "contact for", "phone number of", "call ", "website of"]): return "contact_info"
    if any(kw in query_l for kw in ["book ", "booking ", "availability of", "reserve ", "reservation for"]): return "booking_info"
    if any(kw in query_l for kw in ["how to ", "explain ", "what are the features", "details about"]): return "explanation"
    if any(kw in query_l for kw in ["compare", "difference between"]): return "comparison"
    if any(kw in query_l for kw in ["operating hours", "timings", "when is it open"]): return "operating_hours"
    if any(kw in query_l for kw in ["weather", "time now", "news", "stock price", "meaning of life"]): return "out_of_scope"
    return "general_information"

from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("resources/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

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
    return re.sub(r'(.)\1{2,}', r'\1', text)

def detect_conversational_intent(query):

    query_clean = correct_spelling(query.lower().strip())
    query_clean = collapse_repeated_letters(query_clean)
    logger.debug(f"[IntentDetection] Final cleaned query: {query_clean}")

    # Define intent patterns
    greeting_variants = [
        "hi", "hello", "namaste", "hey", "greetings", "good morning", "good afternoon", "good evening", 
        "good night", "good day", "hiya", "yo", "hey there", "howdy", "salutations"
    ]
    exit_variants = [
        "bye", "goodbye", "see you", "take care", "farewell", "cya", "see ya", "later", 
        "talk to you later", "adios", "catch you later", "gotta go", "until next time",
        "i'm leaving", "that's all", "i'm done", "bye for now", "peace out", "okay bye",
    ]
    smalltalk_variants = [
        "thank you", "thanks", "thx", "ty", "tysm", "ok", "okay", "cool", "sure", "fine", 
        "great", "nice", "good", "awesome", "super", "you there", "u there", "are you there",
        "hello again", "who are you", "how are you", "how’s it going", "what’s up", "wassup", 
        "sup", "yo", "bored", "i’m back", "doing nothing", "tell me something", 
        "you’re cool", "interesting", "love you", "like you", "just checking", "just saying hi",
        "hi again", "you awake?", "you online?", "mood off", "i’m tired", "i’m bored", 
        "anything new?", "say something", "tell me a joke", "reply pls", "pls respond"
    ]
    appreciation_variants = [
        "you are doing good", "good job", "great work", "well done", "very well", 
        "appreciate it", "thanks a lot", "thank you so much", "that’s helpful", 
        "amazing answer", "awesome reply", "you nailed it", "you’re awesome", 
        "you rock", "brilliant", "excellent", "superb", "love that", "fantastic", 
        "mind blowing", "next level", "exactly what I needed", "so quick", "so smart"
    ]
    confirmation_variants = [
        "yes", "yep", "yeah", "sure", "absolutely", "of course", "definitely", 
        "yup", "you got it", "correct", "right", "exactly", "that’s right"
    ]
    negation_variants = [
        "no", "nope", "nah", "not really", "never", "i don't think so", 
        "wrong", "that's incorrect", "nahi", "galat", "bilkul nahi"
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
    
    return None

import re
def is_likely_room_code(token: str) -> bool:
    # Pattern matches like 3a2b, 3-a-2b, 3 a 2b, 7r-2w, etc.
    return bool(re.match(r"^\d+[a-z]([-_\s]?\d+[a-z])?$", token))

def normalize_room_code(token: str) -> str:
    # Replace all separators with dashes
    token = re.sub(r"[-_\s]+", "-", token)
    # Insert dash between digit and letter or letter and digit
    token = re.sub(r"(\d)([a-z])", r"\1-\2", token)
    token = re.sub(r"([a-z])(\d)", r"\1-\2", token)
    return token

def normalize_query(query: str) -> str:
    q = query.lower().strip()

    # Standardize company suffixes
    q = q.replace("pvt ltd", "pvt. ltd.")
    q = q.replace("private limited", "pvt. ltd.")
    q = q.replace("ltd.", "ltd")

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

    # Remove unwanted punctuation (except dash and dot)
    q = re.sub(r"[^\w\s\-\.]", "", q)

    # Normalize spacing
    q = re.sub(r"\s+", " ", q)

    return q.strip()

def canonicalize_entity_value(entity_value):
    value_l = entity_value.lower().strip()
    for canonical, aliases in SYNONYM_MAP.items():
        all_forms = [canonical] + aliases
        if value_l in [a.lower() for a in all_forms]:
            return canonical  # Return the base form
    return entity_value  # No match found

def generate_clarification_suggestions(entities, memory):
    suggestions = []
    recent_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=2)
    if entities.get("rooms") and not entities.get("floors"):
        for entity_info in recent_entities:
            if entity_info["type"] == "floors":
                suggestions.append(f"Did you mean {entities['rooms'][0]} on floor {entity_info['value']}?")
                break
    if not suggestions: suggestions.extend(["Could you specify the floor?", "Can you provide the full name?"])
    return suggestions[:2]

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


def rewrite_query_with_memory(query, memory: ConversationMemory):
    original_query = query.strip()
    rewritten_query = original_query

    # Normalize for fuzzy matching
    query_lower = original_query.lower()

    context_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)

    # Try to resolve vague references by priority (company > room > floor)
    salient_topic_entity = memory.get_last_entity_by_priority()

    # Fallbacks
    if not salient_topic_entity and memory.current_topic:
        salient_topic_entity = memory.current_topic.get("value")
    elif not salient_topic_entity and context_entities:
        salient_topic_entity = context_entities[-1].get("value")


    # === Add follow-up detection here ===
    follow_up_patterns = [
        r"^(and|also|what about|how about)\b.*(email|phone|contact|website|location|address|services|timings|hours)?",
        r"^(their|that|its)\b.*(email|contact|phone|website|location|room|floor|timings|hours|services)",
        r"^(email|phone|contact|website|location|timings|hours|services)\b",
        r"^(email|website|phone|contact|linkedin)\b.*"
        r"^(how to reach|reach out to|connect with|get in touch with)\b",
        r"(do they|are they)\b.*(open|available|operational|provide|offer)",
    ]

    # Direct noun-only follow-ups (1–3 words, like "email?" or "website info")
    if len(query_lower.split()) <= 3 and salient_topic_entity:
        if any(word in query_lower for word in ["email", "website", "linkedin", "phone", "contact"]):
            rewritten_query = f"{salient_topic_entity} — {original_query}"
            logger.info(f"[Coref Fallback] Injected entity into short query: '{original_query}' → '{rewritten_query}'")
            return rewritten_query


    if any(re.search(p, query_lower) for p in follow_up_patterns):
        if salient_topic_entity:
            rewritten_query = f"{salient_topic_entity} — {original_query}"
            return rewritten_query  # early return on follow-up detected

    # Safeguard: don't rewrite unless context is short and vague
    if len(query_lower.split()) < 7 and not any(q_word in query_lower for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"]):
        if not salient_topic_entity and context_entities:
            salient_topic_entity = context_entities[-1].get("value")

        # Extra protection: avoid rewriting if salient entity is a room/floor
        risky_types = ["rooms", "floors", "washrooms"]
        if salient_topic_entity:
            for e in context_entities:
                if e["value"] == salient_topic_entity and e["type"] in risky_types:
                    logger.info(f"[Coref] Skipped coref injection due to risky type: {e}")
                    return rewritten_query  # Skip rewriting

        # Vague coreference phrases
        vague_patterns = [
            r"\bit\b(?!['\w])",
            r"\bthey\b",
            r"\bthem\b",
            r"\btheir\s+(room|company|office|place|location|room)?\b",
            r"\bits\s+(room|company|office|place|location|room)?\b",
            r"\bthis\s+(room|company|office|place|location|room)?\b",
            r"\bthat\s+(room|company|office|place|location|room)?\b",
            r"\bthis one\b",
            r"\bthat one\b",
            r"\bfirst one\b",
            r"\blast one\b",
        ]

        for pattern in vague_patterns:
            if salient_topic_entity:
                replacement = salient_topic_entity
                if re.search(r"\b(their|that|this|its)\b", pattern):
                    replacement += "'s"
                new_query = re.sub(pattern, replacement, rewritten_query, count=1, flags=re.IGNORECASE)
                if new_query != rewritten_query:
                    logger.info(f"[Coref Resolution] Rewrote '{rewritten_query}' → '{new_query}'")
                    rewritten_query = new_query
                    break  # Stop after first match

    if rewritten_query != original_query:
        logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
    return rewritten_query



def handle_small_talk(user_query, memory, session):
    import random
    convo_prompt = f"""
You are a cheerful, helpful assistant at the RNI building in IIT Delhi. The user is being casual, friendly, or saying goodbye. Reply in a varied, natural, warm tone — use emojis sometimes. Avoid repeating yourself.

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
    logger.info(f"--- New Chat Request --- Query: '{user_query}'")
    if "memory" not in session: 
        session["memory"] = ConversationMemory().__dict__
    conv_memory = ConversationMemory()
    try:
        valid_keys = conv_memory.__dict__.keys()
        session_memory_dict = session["memory"]
        loadable_memory = {k: session_memory_dict[k] for k in valid_keys if k in session_memory_dict}
        conv_memory.__dict__.update(loadable_memory)
    except Exception as e:
        logger.error(f"Error loading conversation memory from session: {e}. Initializing new memory.")
        session["memory"] = ConversationMemory().__dict__

    original_user_query = user_query.strip()
    query_lower = original_user_query.lower()
    # 🔍 EARLY: Check for greetings/smalltalk/exit before translation
    convo_intent = detect_conversational_intent(original_user_query)
    entity_keywords = ["room", "office", "company", "building", "floor", "iwayplus", "neo risers", "location", "where", "find"]

    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation"}:
        if not any(word in query_lower for word in entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, session)
    # Language Detection and Translation AFTER early intent checks
    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
    translated_query, detected_input_lang = detect_and_translate(cleaned_query_for_lang_detect)
    processed_query = (
        rewrite_query_with_memory(translated_query, conv_memory) 
        if len(translated_query.split()) < 7 and not any(translated_query.lower().startswith(q_word) for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"]) 
        else translated_query
    )
    extracted_query_entities = nlu_processor.extract_entities(processed_query)
    task_type = nlu_processor.classify_intent(processed_query)
    convo_intent = detect_conversational_intent(processed_query)

    # Avoid triggering small talk handler if user query mentions an entity
    entity_keywords = ["room", "office", "company", "building", "floor", "iwayplus", "neo risers", "location", "where", "find"]
    query_lower = original_user_query.lower()

    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation"}:
        if not any(word in query_lower for word in entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, session)
        else:
            # Treat as mixed: greeting + actual question → combine both
            greeting_resp = handle_small_talk(original_user_query, conv_memory, session)["answer"]
            # We'll append this to final LLM response later


    conv_memory.add_turn(original_user_query, "", extracted_query_entities)
    logger.info(f"Detected task type: {task_type}")

    if task_type == "out_of_scope":
        out_of_scope_response = "I am an assistant for the RNI building and can only answer questions related to its facilities, occupants, and services. How can I help you with that?"
        conv_memory.history[-1]["assistant"] = out_of_scope_response
        session["memory"] = conv_memory.__dict__
        if target_lang_code and target_lang_code != "en": out_of_scope_response = GoogleTranslator(source="en", target=target_lang_code).translate(out_of_scope_response)
        elif detected_input_lang != "en": out_of_scope_response = GoogleTranslator(source="en", target=detected_input_lang).translate(out_of_scope_response)
        return {"answer": out_of_scope_response, "debug_info": {"task_type": task_type}}

    if not GROQ_API_KEY: logger.critical("Groq API key not configured."); return {"answer": "Error: Chat service temporarily unavailable."}
    query_chars = classify_query_characteristics(processed_query)
    response_length_hint = query_chars.get("response_length", "short")
    answer_style, answer_tone = detect_answer_style_and_tone(processed_query)
    logger.info(f"Response hints: length={response_length_hint}, style={answer_style}, tone={answer_tone}")

    retrieved_docs = hybrid_retriever(processed_query, k_simple=6, k_normal=8, k_complex=10)
    if not retrieved_docs:
        logger.warning(f"No documents retrieved for query: {processed_query}")
        clarification_msg = "I couldn't find specific information. "
        suggestions = generate_clarification_suggestions(extracted_query_entities, conv_memory)
        clarification_msg += " ".join(suggestions) if suggestions else "Could you rephrase or provide more details?"
        conv_memory.history[-1]["assistant"] = clarification_msg; session["memory"] = conv_memory.__dict__
        if target_lang_code and target_lang_code != "en": clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en": clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)
        return {"answer": clarification_msg, "related_queries": suggestions if suggestions else []}

    bi_reranked_docs = rerank_documents_bi_encoder(processed_query, retrieved_docs, top_k=6)
    final_docs_for_llm = rerank_documents_cross_encoder(processed_query, bi_reranked_docs, top_k=3)
    # Defensive check: Warn if extracted entities are missing in retrieved context
    entity_terms = set()
    for ent_list in extracted_query_entities.values():
        for val in ent_list:
            val_clean = val.lower().strip()
            # Filter out junk tokens like '##p' or anything too short
            if val_clean and len(val_clean) > 1 and not val_clean.startswith("##"):
                entity_terms.add(val_clean)

    # Log the terms we’re checking
    logger.info(f"[Entity Grounding] Extracted terms: {entity_terms}")

    missing_entities = []
    for term in entity_terms:
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
        session["memory"] = conv_memory.__dict__

        # Translate if needed
        if target_lang_code and target_lang_code != "en":
            clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en":
            clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)

        return {"answer": clarification_msg}



    # Force-inject top BM25 document (if not already present)
    top_bm25_doc = bm25_retriever_func(processed_query, k=1)
    if top_bm25_doc:
        top_doc = top_bm25_doc[0]
        if all(top_doc.page_content.strip() != doc.page_content.strip() for doc in final_docs_for_llm):
            final_docs_for_llm.append(top_doc)
            logger.info("Injected top BM25 doc into LLM context.")
    
    logger.info(f"Final {len(final_docs_for_llm)} documents selected for LLM context.")
    if not final_docs_for_llm and retrieved_docs: final_docs_for_llm = retrieved_docs[:2]; logger.warning("Reranking lost all docs, using top 2 from initial retrieval.")

    context_parts = []
    for i, doc in enumerate(final_docs_for_llm):
        doc_text = f"Source {i+1}:\n{doc.page_content}\n"
        meta_info = {
            "Item Name": doc.metadata.get("item_name"),
            "Company": doc.metadata.get("company_name"),
            "Floor": doc.metadata.get("floor"),
            "Purpose": doc.metadata.get("purpose"),
            "Summary": doc.metadata.get("summary","")[:150] + "..." if doc.metadata.get("summary") else "N/A",
            "Key Tags": ", ".join(doc.metadata.get("tags",[])[:3])
        }
        filtered_meta_info = {k:v for k,v in meta_info.items() if v is not None and v != ""}
        if filtered_meta_info:
            doc_text += "Key Info: " + "; ".join([f"{k}: {v}" for k,v in filtered_meta_info.items()])
        context_parts.append(doc_text)
    extracted_context_str = "\n\n---\n\n".join(context_parts)

    prompt_intro = f"You are a highly advanced, intelligent, and conversational assistant for the Research and Innovation Park (RNI building). Your primary goal is to provide accurate, concise, and relevant information based ONLY on the 'Extracted Context' provided. If the context is insufficient or irrelevant, clearly state that you cannot answer or need more information. Do NOT invent information."
    
    task_instructions = ""
    if task_type in ["location", "location_specific", "location_general"]:
        task_instructions = "Focus on providing location details, floor, tower, and nearby landmarks if available in context. If coordinates are in context and relevant, mention them conceptually (e.g., 'in the north wing')."
    elif task_type == "contact_info":
        task_instructions = "Extract and provide specific contact details like email, phone numbers, or website URLs from the context. If multiple contacts exist, list them clearly."
    elif task_type == "operating_hours":
        task_instructions = "Clearly state the operating hours, including days of the week, start, and end times as found in the context."
    elif task_type in ["explanation", "general_information"]:
        task_instructions = "Provide a comprehensive explanation or description based on the context. If the context has a summary, use it but elaborate with other details if available."
    elif task_type in ["listing_all", "listing_specific"]:
        task_instructions = "List all relevant items based on the query and context. Use bullet points if appropriate."
    elif task_type == "booking_info":
        task_instructions = "Provide booking details, including method, URL, and approval requirements from the context."
    elif task_type == "comparison":
        task_instructions = "Compare the relevant entities based on the context, highlighting differences and similarities."

    prompt_template_str = f"""{prompt_intro}

Strict Rules:
1. Base answers ONLY on 'Extracted Context'. If not in context, say so (e.g., "Based on the provided information, I cannot answer that."). Do NOT use external knowledge.
2. If context is empty/irrelevant, state you lack info.
3. Consider 'Past Conversation History' for implicit details but prioritize current query and context.
4. If query is ambiguous despite context, you can ask one brief clarifying question.
5. Be conversational and helpful. Adapt to the user's likely intent.
6. {task_instructions}

Past Conversation History (Recent Turns):
{{history}}

Extracted Context (Source of Truth):
---
{{context}}
---

User Query: {{input}}
Task Type: {task_type}
Requested Answer Style: {answer_style}
Requested Tone: {answer_tone}
Desired Response Length: {response_length_hint}

Answer:
"""
    chat_history_for_prompt = conv_memory.get_contextual_history_text(num_turns=4)
    llm_input_data = {"input": processed_query, "context": extracted_context_str, "history": chat_history_for_prompt}

    if response_length_hint == "long" or task_type in ["explanation", "comparison", "listing_all"]:
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.4
    elif task_type in ["contact_info", "location"] and response_length_hint == "short":
        groq_llm_model_name, temperature_val = "llama3-8b-8192", 0.1
    else: 
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.25
    logger.info(f"Using Groq model: {groq_llm_model_name} with temperature: {temperature_val}")
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model=groq_llm_model_name, temperature=temperature_val)
    prompt = PromptTemplate.from_template(prompt_template_str)
    runnable_chain = prompt | llm
    final_response_text = "Error: Could not generate a response."
    try:
        ai_message = runnable_chain.invoke(llm_input_data)
        final_response_text = ai_message.content
        logger.info(f"LLM Raw Response Snippet: {final_response_text[:200]}...")
    except Exception as e:
        logger.error(f"Error invoking RAG chain with Groq: {e}")
        final_response_text = "I apologize, but I encountered an issue while processing your request. The context might have been too large."

    conv_memory.history[-1]["assistant"] = final_response_text
    # Prepend greeting if this was a mixed greeting+query
    if convo_intent in {"greeting", "smalltalk"} and 'greeting_resp' in locals():
        final_response_text = f"{greeting_resp}\n\n{final_response_text}"

    session["memory"] = conv_memory.__dict__
    try:
        if target_lang_code and target_lang_code != "en": 
            final_response_text = GoogleTranslator(source="en", target=target_lang_code).translate(final_response_text); 
            logger.info(f"Translated response to {target_lang_code}.")
        elif detected_input_lang != "en" and detected_input_lang is not None: 
            final_response_text = GoogleTranslator(source="en", target=detected_input_lang).translate(final_response_text); 
            logger.info(f"Translated response back to input language {detected_input_lang}.")
    except Exception as e: logger.warning(f"Failed to translate final response: {e}")
    
    processing_time = (datetime.now() - request_start_time).total_seconds()
    logger.info(f"--- Chat Request Completed --- Time: {processing_time:.2f}s")
    debug_info = {
        "task_type": task_type,
        "processed_query": processed_query,
        "detected_input_lang": detected_input_lang,
        "target_lang_code": target_lang_code,
        "response_length_hint": response_length_hint,
        "answer_style": answer_style,
        "answer_tone": answer_tone,
        "llm_model": groq_llm_model_name,
        "retrieved_doc_count_initial": len(retrieved_docs) if retrieved_docs else 0,
        "retrieved_doc_count_final_llm": len(final_docs_for_llm) if final_docs_for_llm else 0,
        "top_doc_titles_for_llm": [doc.metadata.get("item_name", doc.metadata.get("source_doc_id","Unknown")) for doc in final_docs_for_llm] if final_docs_for_llm else [],
        "processing_time_seconds": round(processing_time, 2)
    }
    return {"answer": final_response_text, "debug_info": debug_info}

@app.route("/")
def home(): return render_template("index.html")

@app.route("/refresh_data", methods=["POST"])
def refresh_data_endpoint():
    logger.info("Data refresh request received.")
    global data_loader
    data_loader = RNIDataLoader()
    refresh_faiss_and_bm25()
    logger.info("Data and retrieval models refreshed successfully.")
    return jsonify({"message": "Data and retrieval models refreshed successfully."}), 200

@app.route("/chat", methods=["GET", "POST"])
def chat_endpoint():
    if request.method == "GET":
        user_message = request.args.get("message", "").strip()
    else:
        data = request.get_json()
        user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = chat(user_message)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)