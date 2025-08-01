import logging
import random
import re
from datetime import datetime
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

# Initialize global dependencies
data_loader = HospitalDataLoader()
user_memory_store = InMemoryUserMemoryStore()
nlu_processor = EnhancedNLUProcessor()
retriever = HybridRetriever()
reranker = DocumentReranker()

def classify_query_characteristics(query):
    query_l = query.lower()
    response_length = "short"
    if any(word in query_l for word in ["explain in detail", "everything about"]):
        response_length = "long"
    elif any(word in query_l for word in ["list", "summarize", "overview of"]):
        response_length = "medium"
    return {"response_length": response_length}

def detect_answer_style_and_tone(query):
    query_l = query.lower()
    style = "paragraph"
    tone = "professional_and_helpful"
    if any(word in query_l for word in ["bullet points", "list them", "in bullets", "pointwise", "give points", "step by step", "in steps"]):
        style = "bullet_list"
    if any(word in query_l for word in ["in a table", "tabular format", "as a table", "table format", "make a table", "structured table"]):
        style = "table"
    if any(word in query_l for word in ["friendly", "casual", "informal", "talk like a friend", "light tone", "easy to understand", "simplify it"]):
        tone = "friendly_and_casual"
    if any(word in query_l for word in ["formal", "official statement", "strictly professional", "precise response", "in a formal tone", "business tone"]):
        tone = "formal_and_precise"
    return style, tone

def rewrite_query_with_memory(query, memory: ConversationMemory):
    original_query = query.strip()
    rewritten_query = original_query
    query_lower_normalized = normalize_query(original_query.lower())

    context_entities = memory.get_relevant_entities_from_recent_turns(turns_to_check=3)
    
    salient_topic_entity_value = memory.get_last_entity_by_priority(
        type_priority=["doctors", "departments", "services", "rooms", "hospitals", "buildings"]
    )
    salient_topic_type = None
    if salient_topic_entity_value:
        for entity_type in ["doctors", "departments", "services", "rooms", "hospitals", "buildings"]:
            if memory.last_entity_by_type.get(entity_type) == salient_topic_entity_value:
                salient_topic_type = entity_type
                break
    
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
    follow_up_pattern_str = r"^(and|then|also|what about|how about|tell me more about|more info on)\b.*(" + "|".join(follow_up_keywords) + \
                           r")?|^(their|his|her|its)\b.*(" + "|".join(follow_up_keywords) + \
                           r")|^\b(" + "|".join(follow_up_keywords) + r")\b"
    
    is_short_follow_up_keyword_only = len(original_query.split()) <= 2 and any(kw in query_lower_normalized for kw in follow_up_keywords)

    if salient_topic_entity_value and (re.search(follow_up_pattern_str, query_lower_normalized, re.IGNORECASE) or is_short_follow_up_keyword_only):
        if not re.search(rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', query_lower_normalized, re.IGNORECASE):
            rewritten_query = f"{salient_topic_entity_value} - {original_query}"
            logger.info(f"[Coref Rewrite - Follow-up] Rewrote '{original_query}' → '{rewritten_query}' using salient entity '{salient_topic_entity_value}'")
            return rewritten_query

    if len(original_query.split()) < 5 and salient_topic_entity_value:
        if salient_topic_type in ["floors", "buildings"] or original_query.lower().startswith(("what", "who", "where", "when", "why", "how")):
            if rewritten_query != original_query:
                logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
            return rewritten_query

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
                if not re.search(rf'\b{re.escape(salient_topic_entity_value.split()[0])}\b', query_lower_normalized, re.IGNORECASE):
                    rewritten_query = re.sub(pattern, replacement_text, rewritten_query, count=1, flags=re.IGNORECASE)
                    logger.info(f"[Coref Rewrite - Pronoun] Rewrote '{original_query}' → '{rewritten_query}' replacing '{pronoun}' with '{replacement_text}'")
                    break

    if rewritten_query != original_query:
        logger.info(f"[Coref Final Rewritten Query] '{original_query}' → '{rewritten_query}'")
    return rewritten_query

def handle_small_talk(user_query, memory, user_id):
    convo_prompt = (
        f"You are a cheerful, friendly assistant at the AIIMS building in Jammu.\n"
        f"The user is being casual, greeting, or saying goodbye.\n"
        f"Reply naturally and warmly. Keep it under 20 words. You can use a single friendly emoji. Never repeat yourself.\n"
        f"\nUser: {user_query}\nAssistant:"
    )

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.85)
        response = llm.invoke(convo_prompt).content.strip()
    except Exception as e:
        logger.error(f"Error in small talk handler: {e}")
        response = random.choice([
            "Hey there! 👋 How can I assist you at AIIMS Jammu today?",
            "Hi! 😊 I'm here to help with anything related to the hospital.",
            "Hello! Just ask if you need directions, doctor info, or anything else!",
            "Hey! Need any help with appointments or departments?"
        ])

    memory.add_turn(user_query, response, extracted_entities_map={})
    user_memory_store.save(user_id, memory)
    return {"answer": response}

def chat(user_query: str, user_id: str):
    request_start_time = datetime.now()
    logger.info(f"--- New Chat Request (Hospital) --- User ID: {user_id} | Query: '{user_query}'")

    conv_memory = user_memory_store.get(user_id)
    original_user_query = user_query.strip()
    query_lower_raw = original_user_query.lower()

    convo_intent = detect_conversational_intent(original_user_query)
    hospital_entity_keywords = [
        "room", "opd", "icu", "ward", "doctor", "dr", "nurse", "staff", "physician", "consultant", "specialist",
        "department", "cardiology", "neurology", "oncology", "pediatrics", "radiology", "surgery", "clinic",
        "service", "x-ray", "mri", "scan", "test", "appointment", "treatment", "procedure",
        "hospital", "aiims", "building", "floor", "location", "find", "where", "contact", "phone", "email",
        "availability", "schedule", "hours", "timings"
    ]
    
    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(keyword in query_lower_raw for keyword in hospital_entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, user_id)

    cleaned_query_for_lang_detect, target_lang_code = detect_target_language_for_response(original_user_query)
    translated_query, detected_input_lang = detect_and_translate(cleaned_query_for_lang_detect)
    processed_query = (
        rewrite_query_with_memory(translated_query, conv_memory)
        if len(translated_query.split()) < 7 and not any(translated_query.lower().startswith(q_word) for q_word in ["what", "who", "where", "when", "why", "how", "list", "explain", "compare"])
        else translated_query
    )

    extracted_query_entities = nlu_processor.extract_entities(processed_query)
    extracted_query_entities = clean_extracted_entities(extracted_query_entities)

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

    known_tags = set(data_loader.get_all_metadata_tags())

    for key, val in extracted_query_entities.items():
        if isinstance(val, str) and val.lower() in known_tags:
            logger.info(f"[Tag Match] Entity '{val}' matched known metadata tag")
            extracted_query_entities[f"matched_tag__{key}"] = val.lower()
        elif isinstance(val, list):
            for subval in val:
                if isinstance(subval, str) and subval.lower() in known_tags:
                    logger.info(f"[Tag Match] List Entity '{subval}' matched known tag")
                    extracted_query_entities.setdefault(f"matched_tag__{key}", []).append(subval.lower())

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

    task_type = nlu_processor.classify_intent(processed_query)
    convo_intent = detect_conversational_intent(processed_query)

    if convo_intent in {"greeting", "exit", "smalltalk", "appreciation", "confirmation", "negation", "help"}:
        if not any(word in query_lower_raw for word in hospital_entity_keywords):
            return handle_small_talk(original_user_query, conv_memory, user_id)
        else:
            greeting_resp = handle_small_talk(original_user_query, conv_memory, user_id)["answer"]

    conv_memory.add_turn(original_user_query, "", extracted_query_entities)
    user_memory_store.save(user_id, conv_memory)
    logger.info(f"Detected task type (hospital): {task_type}")

    if task_type == "out_of_scope":
        out_of_scope_response = "I am an assistant for AIIMS Jammu hospital and can only answer questions related to its facilities, departments, doctors, services, and appointments. How can I help you with that?"
        conv_memory.history[-1]["assistant"] = out_of_scope_response
        user_memory_store.save(user_id, conv_memory)
        if target_lang_code and target_lang_code != "en":
            out_of_scope_response = GoogleTranslator(source="en", target=target_lang_code).translate(out_of_scope_response)
        elif detected_input_lang != "en":
            out_of_scope_response = GoogleTranslator(source="en", target=detected_input_lang).translate(out_of_scope_response)
        return {"answer": out_of_scope_response, "debug_info": {"task_type": task_type}}

    if not GROQ_API_KEY:
        logger.critical("Groq API key not configured.")
        return {"answer": "Error: Chat service temporarily unavailable."}

    query_chars = classify_query_characteristics(processed_query)
    response_length_hint = query_chars.get("response_length", "short")
    answer_style, answer_tone = detect_answer_style_and_tone(processed_query)
    logger.info(f"Response hints (hospital): length={response_length_hint}, style={answer_style}, tone={answer_tone}")

    retrieved_docs = retriever.hybrid_search(processed_query, k_simple=6, k_normal=10, k_complex=15)
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

    bi_reranked_docs = reranker.rerank_documents_bi_encoder(processed_query, retrieved_docs, top_k=6)
    final_docs_for_llm = reranker.rerank_documents_cross_encoder(processed_query, bi_reranked_docs, top_k=3)

    doctor_candidates = extracted_query_entities.get("doctors", []) + extracted_query_entities.get("persons", [])
    doctor_candidates = [d for d in doctor_candidates if len(d.split()) >= 2]
    query_doctor_name = doctor_candidates[0] if doctor_candidates else extract_doctor_name(processed_query)

    if query_doctor_name:
        response = get_doctor_by_name(query_doctor_name, final_docs_for_llm)
        if response:
            logger.info(f"[Doctor Match] Structured match found for: {query_doctor_name}")
            return {"answer": response}

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
        user_memory_store.save(user_id, conv_memory)
        if target_lang_code and target_lang_code != "en":
            clarification_msg = GoogleTranslator(source="en", target=target_lang_code).translate(clarification_msg)
        elif detected_input_lang != "en":
            clarification_msg = GoogleTranslator(source="en", target=detected_input_lang).translate(clarification_msg)
        return {"answer": clarification_msg}

    top_bm25_docs_for_injection = retriever.bm25_retrieve(processed_query, k=1)
    if top_bm25_docs_for_injection:
        top_bm25_doc = top_bm25_docs_for_injection[0]
        if all(top_bm25_doc.page_content.strip() != doc.page_content.strip() for doc in final_docs_for_llm):
            final_docs_for_llm.append(top_bm25_doc)
            logger.info("Injected top BM25 doc into LLM context.")

    logger.info(f"Final {len(final_docs_for_llm)} documents selected for LLM context (hospital).")
    if not final_docs_for_llm and retrieved_docs:
        final_docs_for_llm = retrieved_docs[:3]
        logger.warning("Reranking resulted in zero documents. Using top 3 from initial hybrid retrieval for LLM.")

    context_parts = []
    for i, doc in enumerate(final_docs_for_llm):
        doc_text = f"Source Document {i+1}:\n{doc.page_content}\n"
        meta_info = {
            "Hospital": doc.metadata.get("hospital_name"),
            "Building": doc.metadata.get("building_name"),
            "Floor": doc.metadata.get("floor"),
            "Room Name": doc.metadata.get("room_name"),
            "Room Number": doc.metadata.get("room_number"),
            "Associated Depts": ", ".join(ensure_list(doc.metadata.get("associated_departments", []))[:2]),
            "Associated Doctors": ", ".join(ensure_list(doc.metadata.get("associated_doctors", []))[:2]),
            "Key Services": (", ".join(ensure_list(doc.metadata.get("services_directly_offered", []))[:2]) or
                             ", ".join(ensure_list(doc.metadata.get("department_related_services", []))[:2])),
            "Doc ID": doc.metadata.get("source_doc_id")
        }
        filtered_meta_info = {k: v for k, v in meta_info.items() if v is not None and v != ""}
        if filtered_meta_info:
            doc_text += "Key Metadata: " + "; ".join([f"{k}: {v}" for k, v in filtered_meta_info.items()])
        context_parts.append(doc_text)
    extracted_context_str = "\n\n---\n\n".join(context_parts)

    prompt_intro = f"You are a highly advanced, intelligent, and conversational AI assistant for AIIMS Jammu Building. Your primary goal is to provide accurate, concise, and relevant information based ONLY on the 'Extracted Context' provided. If the context is insufficient or irrelevant, clearly state that you cannot answer or need more information. Do NOT invent information or use external knowledge."

    task_instructions = ""
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
    elif task_type == "booking_info":
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

    if response_length_hint == "long" or task_type in ["explanation", "comparison", "listing_all"] or "complex" in detect_query_complexity(processed_query):
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.4
    elif task_type in ["contact_info", "location", "doctor_availability"] and response_length_hint == "short":
        groq_llm_model_name, temperature_val = "llama3-8b-8192", 0.15
    else:
        groq_llm_model_name, temperature_val = "llama3-70b-8192", 0.25
    logger.info(f"Using Groq model (hospital): {groq_llm_model_name} with temperature: {temperature_val}")

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
        final_response_text = "I apologize, but I encountered an issue while processing your request. The context might have been too large."

    conv_memory.history[-1]["assistant"] = final_response_text
    if convo_intent in {"greeting", "smalltalk"} and 'greeting_resp' in locals():
        final_response_text = f"{greeting_resp}\n\n{final_response_text}"

    user_memory_store.save(user_id, conv_memory)

    try:
        if target_lang_code and target_lang_code != "en":
            final_response_text = GoogleTranslator(source="en", target=target_lang_code).translate(final_response_text)
            logger.info(f"Translated response to {target_lang_code}.")
        elif detected_input_lang != "en" and detected_input_lang is not None:
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
        "final_doc_ids_for_llm": [doc.metadata.get("source_doc_id", "Unknown") for doc in final_docs_for_llm] if final_docs_for_llm else [],
        "processing_time_seconds": round(processing_time, 2),
        "conversational_intent": convo_intent,
        "extracted_entities": extracted_query_entities,
        "query_complexity": classify_query_characteristics(processed_query),
        "missing_entities_in_docs": missing_entities
    }
    return {"answer": final_response_text, "debug_info": debug_info}