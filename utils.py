import re
import logging
from deep_translator import GoogleTranslator
from langdetect import detect
from rapidfuzz import fuzz
from symspellpy import SymSpell, Verbosity
from data_loader import HospitalDataLoader
from memory import ConversationMemory
from config import SYNONYM_MAP

logger = logging.getLogger(__name__)

data_loader = HospitalDataLoader()

def ensure_list(val):
    if isinstance(val, list):
        return val
    elif val:
        return [val]
    return []

def format_operating_hours(hours_data):
    if not hours_data:
        return "N/A"

    order = ["mondayToFriday", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
    parts = []
    
    for day in order:
        if day not in hours_data:
            continue
        
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
    if not channels_list:
        return "N/A"

    parts = []
    
    for ch in channels_list:
        ch_type = ch.get('type', 'Unknown')
        description = ch.get('description', '')
        contact = ch.get('contact', {})
        
        contact_parts = []
        if contact.get('phone'):
            contact_parts.append(f"Phone: {contact['phone']}")
        if contact.get('email'):
            contact_parts.append(f"Email: {contact['email']}")
        if contact.get('website'):
            contact_parts.append(f"Profile website: {contact['website']}")

        contact_str = ", ".join(contact_parts) if contact_parts else "N/A"
        
        channel_info = f"{ch_type}"
        if description:
            channel_info += f" ({description})"
        channel_info += f": {contact_str}"

        op_hours = format_operating_hours(ch.get('operatingHours', {}))
        if op_hours != "N/A":
            channel_info += f" [Hours: {op_hours}]"

        parts.append(channel_info)

    return ". ".join(parts)

def format_doctor_availability(availability_data):
    if not availability_data:
        return "Not specified"
    
    days = ensure_list(availability_data.get("days", []))
    time = availability_data.get("time", "Not specified")
    
    if not days:
        return f"Time: {time}" if time != "Not specified" else "Availability: Not specified"
    
    return f"Days: {', '.join(days)}; Time: {time}"

def format_contact_info(contact_data):
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

def detect_task_type_rule_based(query):
    query_l = query.lower()
    if any(kw in query_l for kw in ["compare", "difference between", "vs ", "versus", "vs."]): return "compare"
    if any(kw in query_l for kw in ["book appointment", "appointment with", "reserve slot", "schedule visit", "want to meet", "book a slot", "fix appointment"]): return "booking_info"
    if any(kw in query_l for kw in ["list all", "show all doctors", "all departments", "every service"]): return "listing_all"
    if any(kw in query_l for kw in ["list services", "list some", "list three", "overview of", "show some doctors", "some departments", "few treatments", "give examples of"]): return "listing_specific"
    if any(kw in query_l for kw in ["where is", "location of", "find near", "how to reach", "direction to", "which floor", "room number", "nearest", "reach the hospital", "hospital location"]): return "location"
    if any(kw in query_l for kw in ["email of", "contact for", "phone number", "call dr", "how to contact", "contact number", "mobile number", "hospital email", "reach dr", "website for hospital"]): return "contact_info"
    if any(kw in query_l for kw in ["how to", "explain", "procedure for", "what are symptoms", "symptoms of", "treatment for", "details about", "how is", "explanation", "tell me about", "elaborate", "how does it work"]): return "explanation"
    if any(kw in query_l for kw in ["operating hours", "timings", "when is opd open", "doctor schedule", "visiting hours", "opd hours", "working hours", "closing time"]): return "operating_hours"
    if any(kw in query_l for kw in ["doctor availability", "is dr available", "dr schedule", "when is dr", "available doctor", "which doctor is available"]): return "doctor_availability"
    if any(kw in query_l for kw in ["department of", "info on", "tell me about cardiology", "cardiology services", "what is neurology", "specialty of", "department details", "which department handles"]): return "department_info"
    if any(kw in query_l for kw in ["service offered", "do you have", "available services", "is mri available", "ct scan facility", "x-ray available", "cost of service", "price of", "charges for", "ambulance available", "test cost"]): return "service_info"
    if any(kw in query_l for kw in ["weather", "time now", "news", "stock price", "meaning of life", "who are you", "what is your name", "are you human"]): return "out_of_scope"
    return "general_information"

def extract_entities_rule_based(query):
    query_lower = query.lower()
    entities = {
        "hospitals": [], "buildings": [], "floors": [], "rooms": [],
        "departments": [], "doctors": [], "services": [],
        "lifts": [], "stairs": [], "washrooms": [], "general_terms": []
    }

    hospital_matches = re.findall(r'\b(aiims(?:\s+\w+)?|jammu|\s+healthcare|city\s+hospital)\b', query_lower, re.IGNORECASE)
    for m in hospital_matches:
        entities["hospitals"].append(m.strip())
    if not entities["hospitals"] and "hospital" in query_lower:
        entities["hospitals"].append("hospital")

    building_matches = re.findall(r'\b(block\s*[\w\d-]+|building\s*[\w\d-]*|tower\s*[\w\d-]*|wing\s*[\w\d-]+|diagnostic\s*block)\b', query_lower, re.IGNORECASE)
    for m in building_matches:
        entities["buildings"].append(m.strip())

    floor_matches = re.findall(r'(?:floor|level|flr)\s*(\d+[-\w]*\b)|(\b\d+)(?:st|nd|rd|th)?\s*(?:floor|level|flr)|(ground\s*floor|gf\b)', query_lower, re.IGNORECASE)
    for m_tuple in floor_matches:
        val = next(filter(None, m_tuple), None)
        if val:
            if "ground floor" in val or "gf" == val:
                entities["floors"].append("0")
            else:
                entities["floors"].append(re.sub(r'[^\d\w-]', '', val))

    room_matches = re.findall(r'\b(?:room|rm|cabin|opd)\s*([\w\d-]+)\b|(\b\d+[A-Za-z]?-?\d*[A-Za-z]?\b(?!\s*floor|\s*st|\s*nd|\s*rd|\s*th|\s*am|\s*pm))', query_lower, re.IGNORECASE)
    for m_tuple in room_matches:
        val = next(filter(None, m_tuple), None)
        if val and len(val) > 1:
            entities["rooms"].append(val.strip())

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
            if keyword in ["opd", "icu", "emergency", "pharmacy", "pathology", "radiology"]:
                entities["departments"].append(keyword.lower())

    department_keywords = [
        "cardiology", "neurology", "oncology", "pediatrics", "paediatrics", "radiology", "surgery", "opd", "outpatient",
        "emergency", "casualty", "icu", "intensive care", "orthopedics", "ortho", "gynecology", "gynaecology",
        "dermatology", "ent", "ear nose throat", "urology", "psychiatry", "pathology", "laboratory", "lab",
        "pharmacy", "physiotherapy", "anesthesia", "dental", "opthalmology", "eyes"
    ]
    for dept_kw in department_keywords:
        if re.search(rf'\b{dept_kw}\b', query_lower, re.IGNORECASE):
            entities["departments"].append(dept_kw)
    dept_matches = re.findall(r'\b(' + '|'.join(department_keywords) + r')\s*(?:department|dept|clinic|unit|ward|center)\b', query_lower, re.IGNORECASE)
    for m in dept_matches:
        entities["departments"].append(m.strip())

    prefix_matches = re.findall(r'\b(?:dr\.?|doctor)\s+([a-z][a-z\s\.-]{2,})\b', query_lower, re.IGNORECASE)
    for name in prefix_matches:
        cleaned = re.sub(r'\b(department|clinic|hospital|ward|unit)\b', '', name, flags=re.IGNORECASE).strip()
        if len(cleaned.split()) >= 1 and len(cleaned) > 3:
            entities["doctors"].append(cleaned.title())

    if data_loader and data_loader.all_known_entities:
        known_doctors = data_loader.all_known_entities.get("doctors", [])
        for doc_name in known_doctors:
            doc_name_clean = re.sub(r'\bdr\.?\s*', '', doc_name, flags=re.IGNORECASE).strip()
            if all(part in query_lower for part in doc_name_clean.lower().split()):
                entities["doctors"].append(doc_name)
                continue
            score = fuzz.token_sort_ratio(query_lower, doc_name_clean.lower())
            if score >= 80:
                entities["doctors"].append(doc_name)

    if not entities["doctors"]:
        name_like_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', query, re.UNICODE)
        for name in name_like_matches:
            if len(name.split()) >= 2 and len(name) > 5:
                entities["doctors"].append(name.strip())

    service_keywords = ["x-ray", "mri", "ct scan", "ultrasound", "ecg", "blood test", "consultation", "therapy", "checkup", "vaccination", "dialysis", "angiography", "biopsy", "endoscopy", "echo", "tmt", "blood sugar test", "cbc", "serology"]
    for svc_kw in service_keywords:
        if re.search(rf'\b{re.escape(svc_kw)}\b', query_lower, re.IGNORECASE):
            entities["services"].append(svc_kw)

    if re.search(r'\b(lift|elevator)\b', query_lower):
        entities["lifts"].append("lift")
    if re.search(r'\b(stairs|staircase)\b', query_lower):
        entities["stairs"].append("stairs")
    if re.search(r'\b(washroom|toilet|restroom|lavatory|wc)\b', query_lower):
        entities["washrooms"].append("washroom")

    if data_loader and data_loader.all_known_entities:
        for entity_type, known_list in data_loader.all_known_entities.items():
            if entity_type in entities:
                for known_item in known_list:
                    if re.search(rf'\b{re.escape(known_item)}\b', query_lower, re.IGNORECASE):
                        entities[entity_type].append(known_item)
                    for syn_keyword, syn_list in SYNONYM_MAP.items():
                        if known_item.lower() == syn_keyword.lower() or known_item.lower() in [s.lower() for s in syn_list]:
                            for form in [syn_keyword] + syn_list:
                                if re.search(rf'\b{re.escape(form)}\b', query_lower, re.IGNORECASE):
                                    entities[entity_type].append(known_item)
                                    break
                            break

    for k in entities:
        entities[k] = sorted(list(set(e.strip() for e in entities[k] if e and len(e.strip()) > 1)))

    if "dr" in entities["doctors"] and len(entities["doctors"]) > 1:
        entities["doctors"].remove("dr")
    if "doctor" in entities["doctors"] and len(entities["doctors"]) > 1:
        entities["doctors"].remove("doctor")
    logger.info(f"[RuleBased NER] Final extracted entities for '{query}': {entities}")
    return entities

def extract_doctor_name(text: str) -> str:
    try:
        match = re.search(r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        return match.group(0).strip() if match else ""
    except Exception as e:
        logger.error(f"[extract_doctor_name error]: {e}")
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
    norm = lambda x: x.lower().replace("dr ", "").strip()
    for doc in docs:
        for doctor in doc.metadata.get("doctor_profiles", []):
            if norm(query_name) in norm(doctor["name"]):
                return format_doctor_response(doctor)
    return None

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    sym_spell.load_dictionary("resources/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
except Exception as e:
    logger.warning(f"Could not load spelling dictionary: {e}. Spelling correction might be affected.")

def correct_spelling(text, verbose=False):
    suggestions = sym_spell.lookup_compound(text.lower(), max_edit_distance=2)
    if suggestions:
        if verbose:
            logger.debug(f"[SpellCheck] Suggestions for '{text}':")
            for s in suggestions:
                logger.debug(f"  - {s.term} (distance: {s.distance}, count: {s.count})")
        corrected = suggestions[0].term
        if verbose:
            logger.debug(f"[SpellCheck] '{text}' → '{corrected}'")
        return corrected
    if verbose:
        logger.debug(f"[SpellCheck] No correction for '{text}'")
    return text

def collapse_repeated_letters(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1', text)

def detect_conversational_intent(query):
    query_corrected_initial = correct_spelling(query)
    query_clean = collapse_repeated_letters(query_corrected_initial.lower().strip())

    greeting_variants = [
        "hi", "hello", "namaste", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "good night", "good day", "hiya", "yo", "hey there", "howdy", "salutations", "sup"
    ]
    exit_variants = [
        "bye", "goodbye", "see you", "take care", "farewell", "cya", "see ya", "later",
        "talk to you later", "adios", "catch you later", "gotta go", "until next time",
        "i'm leaving", "that's all", "i'm done", "bye for now", "peace out", "okay bye", "exit"
    ]
    smalltalk_variants = [
        "how are you", "how’s it going", "what’s up", "wassup",
        "bored", "i’m back", "doing nothing", "tell me something",
        "interesting", "just checking", "just saying hi",
        "hi again", "you awake?", "you online?", "mood off", "i’m tired", "i’m bored",
        "anything new?", "say something", "tell me a joke", "reply pls", "pls respond",
        "ok", "okay", "cool", "sure", "fine", "great", "nice", "good", "awesome", "super"
    ]
    appreciation_variants = [
        "thank you", "thanks", "thx", "ty", "tysm",
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

def is_likely_room_code(token: str) -> bool:
    return bool(re.match(r"^\d+[a-z]([-_\s]?\d+[a-z])?$", token, re.IGNORECASE)) or \
           bool(re.match(r"^[A-Za-z]?\d{2,}[A-Za-z]?$", token))

def normalize_room_code(token: str) -> str:
    token = re.sub(r"[-_\s]+", "", token)
    token = re.sub(r"(\d)([a-z])", r"\1-\2", token)
    token = re.sub(r"([a-z])(\d)", r"\1-\2", token)
    return token.upper()

def normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = collapse_repeated_letters(q)
    q = q.replace("dept.", "department")
    q = q.replace("dr.", "doctor")
    q = re.sub(r"\b(opd|room|rm|cabin|ward|office|icu)\s*(\d+[a-z]?)", r"\1-\2", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(\d+[a-z]?)\s*(opd|room|rm|cabin|ward|office|icu)", r"\2-\1", q, flags=re.IGNORECASE)
    q = re.sub(r"lift\s*lobby[-\s]*(\d+)", r"lift lobby \1", q)
    q = re.sub(r"(.)\1{2,}", r"\1", q)
    tokens = q.split()
    normalized_tokens = []
    for token in tokens:
        if is_likely_room_code(token):
            normalized_tokens.append(normalize_room_code(token))
        else:
            normalized_tokens.append(token)
    q = " ".join(normalized_tokens)
    q = re.sub(r"[^\w\s\-\.@]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def canonicalize_entity_value(entity_value):
    value_l = entity_value.lower().strip()
    for canonical, aliases in SYNONYM_MAP.items():
        all_forms = [canonical.lower()] + [a.lower() for a in aliases]
        if value_l in all_forms:
            return canonical
    return entity_value

def generate_clarification_suggestions(entities, memory: ConversationMemory):
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

def detect_target_language_for_response(query):
    language_map = {"hindi": "hi", "punjabi": "pa", "tamil": "ta", "telugu": "te", "kannada": "kn", "marathi": "mr", "bengali": "bn", "urdu": "ur", "gujarati": "gu", "malayalam": "ml", "english": "en", "spanish": "es", "french": "fr", "german": "de", "russian": "ru"}
    query_lower = query.lower()
    for lang_name, lang_code in language_map.items():
        if re.search(rf'\bin\s+{re.escape(lang_name)}\b', query_lower):
            cleaned_query = re.sub(rf'\s*\bin\s+{re.escape(lang_name)}\b', '', query_lower, flags=re.IGNORECASE).strip()
            logger.info(f"Detected target response language: {lang_name} ({lang_code}). Cleaned query: '{cleaned_query}'")
            return cleaned_query, lang_code
    return query, None

def detect_query_complexity(query):
    query_lower = query.lower()
    if any(conj in query_lower for conj in ["and", "or", "but also", "as well as", "both" " all similar"]) and len(query.split()) > 7: 
        return "complex"
    if any(word in query_lower for word in ["list all services", "all doctors in department", "explain treatment options", "compare procedures", "compare", "list all", "list some"]): 
        return "complex"
    if len(query.split()) <= 5 and any(q_word in query_lower for q_word in ["where is", "dr. email", "phone for", "contact details of", "who is", "what is"]): 
        return "simple"
    return "normal"

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