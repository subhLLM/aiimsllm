import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for Hospital Data
FAISS_INDEX_PATH = "rag_aimms_jammu"
HOSPITAL_MODEL_JSON_PATH = "hospital_building.json"
QA_PAIRS_JSON_PATH = "jammu_qa_pairs_cleaned.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALLOWED_FILES = {os.path.basename(HOSPITAL_MODEL_JSON_PATH),
                 os.path.basename(QA_PAIRS_JSON_PATH)}

# Synonym map for entity canonicalization
SYNONYM_MAP = {
    "elevator": ["lift"], "lift": ["elevator"],
    "toilet": ["washroom", "restroom", "lavatory", "wc"],
    "washroom": ["toilet", "restroom", "lavatory", "wc"],
    "restroom": ["toilet", "washroom", "lavatory", "wc"],
    "stairs": ["staircase"], "staircase": ["stairs"],
    "contactno": ["contact number", "phone no", "phone number", "contact"],
    "timings": ["operating hours", "open hours", "availability", "schedule"],
    "operating hours": ["timings", "open hours", "availability", "schedule"],
    "floor 0": ["ground floor", "gf"], "ground floor": ["floor 0"],
    # Hospital specific
    "doctor": ["dr", "dr.", "physician", "consultant", "specialist", "professor"],
    "dr.": ["doctor", "physician", "consultant", "specialist", "professor"],
    "opd": ["outpatient department", "out-patient department", "clinic", "department"],
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