import logging
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
import torch
import re
from utils import detect_task_type_rule_based, extract_entities_rule_based

logger = logging.getLogger(__name__)

class NLUProcessor:
    def __init__(self):
        try:
            self.intent_labels = [
                "location", "contact_info", "booking_info", "explanation", "comparison",
                "operating_hours", "listing_all", "listing_specific", "out_of_scope", "general_information",
                "doctor_availability", "department_info", "service_info"
            ]
            self.intent_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.intent_label_embeddings = self.intent_encoder.encode(self.intent_labels, convert_to_tensor=True)

            self.ner_pipeline_primary = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER", use_fast=True),
                aggregation_strategy="simple"
            )
            
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
                "lifts": [], "stairs": [], "washrooms": [],
                "general_terms": [], "misc": []
            }
            for entity in ner_results:
                entity_type = entity.get("entity_group", "").upper()
                value = entity.get("word", "").strip()

                if entity_type == "LOC":
                    if "hospital" in value.lower() or "aiims" in value.lower():
                        entities["hospitals"].append(value)
                    elif "building" in value.lower() or "diagnostic" in value.lower() or "wing" in value.lower() or "block" in value.lower() or "tower" in value.lower():
                        entities["buildings"].append(value)
                    elif "floor" in value.lower() or re.match(r'\d+', value.split()[0] if value else ""):
                        entities["floors"].append(value)
                    elif "room" in value.lower() or re.match(r'\w*\d+\w*', value):
                        entities["rooms"].append(value)
                    elif "lift" in value.lower():
                        entities["lifts"].append(value)
                    elif "stair" in value.lower():
                        entities["stairs"].append(value)
                    elif "washroom" in value.lower() or "restroom" in value.lower() or "toilet" in value.lower() or "bathroom" in value.lower():
                        entities["washrooms"].append(value)
                    else:
                        entities["general_terms"].append(value)
                elif entity_type == "ORG":
                    if "hospital" in value.lower() or "aiims" in value.lower():
                        entities["hospitals"].append(value)
                    elif any(dept_kw in value.lower() for dept_kw in ["department", "cardiology", "anesthesiology", "opd", "clinic", "radiology", "surgery"]):
                        entities["departments"].append(value)
                    else:
                        entities["hospitals"].append(value)
                elif entity_type == "PER":
                    if value.lower().startswith("dr") or "doctor" in value.lower():
                        entities["doctors"].append(value)
                    else:
                        entities["doctors"].append(value)
                elif entity_type == "MISC":
                    entities["general_terms"].append(value)
                else:
                    entities["misc"].append(value)

            for k in entities:
                entities[k] = sorted(list(set(entities[k])))
            return entities

        try:
            if self.ner_pipeline_primary:
                primary_results = self.ner_pipeline_primary(query)
                entities = process_ner_results(primary_results)
                logger.info(f"[NER Primary] Extracted entities: {entities}")

                if (not entities.get("doctors") and not entities.get("departments")) and self.ner_pipeline_fallback:
                    logger.info(f"[NER Primary] Sparse entities, trying fallback NER.")
                    fallback_results = self.ner_pipeline_fallback(query)
                    fallback_entities = process_ner_results(fallback_results)
                    logger.info(f"[NER Fallback] Extracted entities: {fallback_entities}")
                    for key, value_list in fallback_entities.items():
                        if value_list and not entities.get(key):
                            entities[key] = value_list
                        elif value_list:
                            entities[key] = sorted(list(set(entities[key] + value_list)))
                
                if not any(entities.get(k) for k in ["doctors", "departments", "rooms", "hospitals", "washrooms", "offices", "wards", "opd", "clinics"]):
                    logger.info(f"[NER] Still sparse, augmenting with rule-based NER.")
                    rule_based_entities = extract_entities_rule_based(query)
                    for key, value_list in rule_based_entities.items():
                        if value_list and not entities.get(key):
                            entities[key] = value_list
                        elif value_list:
                            entities[key] = sorted(list(set(entities[key] + value_list)))

                from data_loader import HospitalDataLoader
                data_loader = HospitalDataLoader()
                if data_loader and data_loader.all_known_entities:
                    known_doctors = data_loader.all_known_entities.get("doctors", [])
                    query_clean = re.sub(r'[^a-zA-Z\s]', '', query).strip().lower()
                    for doc_name in known_doctors:
                        doc_name_clean = doc_name.lower()
                        if query_clean in doc_name_clean or doc_name_clean in query_clean:
                            if doc_name not in entities["doctors"]:
                                entities["doctors"].append(doc_name)

                    entities["doctors"] = [d for d in entities["doctors"] if len(d) > 4 and not d.startswith("##")]

                logger.info(f"[NER Final] Extracted entities: {entities}")
                return entities
        except Exception as e:
            logger.error(f"Primary or Fallback NER failed: {e}")

        logger.info(f"[NER] All NER pipelines failed or skipped, using only rule-based NER.")
        return extract_entities_rule_based(query)

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
        intent = super().classify_intent(query)
        query_lower = query.lower()
        
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