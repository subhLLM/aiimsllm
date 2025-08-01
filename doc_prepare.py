import logging
import re
from langchain_core.documents import Document
from data_loader import HospitalDataLoader
from utils import format_operating_hours, format_response_channels, ensure_list

logger = logging.getLogger(__name__)

data_loader = HospitalDataLoader()

def prepare_documents():
    if not data_loader.hospital_data and not data_loader.qa_data:
        logger.error("No data loaded. Cannot prepare documents.")
        return []

    documents = []
    if data_loader.hospital_data:
        documents.extend(_prepare_hospital_documents())
    
    if data_loader.qa_data:
        documents.extend(_prepare_qa_documents())
    
    logger.info(f"Prepared {len(documents)} total documents for FAISS index.")
    return documents

def _prepare_hospital_documents():
    documents = []
    for item_index, item_data in enumerate(data_loader.hospital_data):
        content_parts = []
        metadata_payload = {
            "source_doc_id": item_data.get("id", f"hospital_item_{item_index}"),
            "document_type": "hospital_data",
            "type": item_data.get("physical", {}).get("type", "UnknownType").lower()
        }

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

            entity_hours = format_operating_hours(entity.get("operatingHours", {}))
            content_parts.append(f"Entity Hours: {entity_hours}.")

            services = ", ".join(ensure_list(entity.get("servicesOffered", []))) or "N/A"
            content_parts.append(f"Services: {services}.")

        avail = func.get("availability", {})
        avail_str = (
            f"Open: {', '.join(ensure_list(avail.get('daysOpen', [])))}. "
            f"Hours: {avail.get('startTime', 'N/A')} - {avail.get('endTime', 'N/A')}."
        ) if avail else "N/A"
        content_parts.append(f"Availability: {avail_str}")

        booking = func.get("booking", {})
        if booking.get("enabled"):
            book_str = (
                f"Method: {booking.get('method', 'N/A')}, URL: {booking.get('url', 'N/A')}. "
                f"Approval: {booking.get('approvalRequired', False)}. "
                f"Notes: {booking.get('notes', 'N/A')[:100]}..."
            )
            content_parts.append(f"Booking: {book_str}")

        response = format_response_channels(func.get("responseChannels", []))
        content_parts.append(f"Response Channels: {response}.")

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

        amenities = ", ".join(ensure_list(acc.get("amenities", []))) or "N/A"
        content_parts.append(f"Amenities: {amenities}.")
        metadata_payload["amenities_summary"] = amenities[:100]

        images = item_data.get("media", {}).get("images", [])
        image_urls = ", ".join(img.get("url") for img in images if img.get("url"))
        content_parts.append(f"Images: {image_urls or 'N/A'}.")

        status = item_data.get("status", {})
        content_parts.append(
            f"Status: {'Op' if status.get('operational') else 'NonOp'}. "
            f"Maint: {status.get('underMaintenance', False)}. "
            f"Insp: {status.get('lastInspected', 'N/A')}."
        )

        meta = item_data.get("metadata", {})
        tags = ", ".join(ensure_list(meta.get("tags", []))) or "N/A"
        content_parts.append(f"Tags: {tags}.")
        metadata_payload["tags"] = ensure_list(meta.get("tags", []))[:5]

        summary = meta.get("summary", "No summary.")[:200]
        content_parts.append(f"Summary: {summary}...")
        metadata_payload["summary"] = summary
        metadata_payload["priority"] = meta.get("priority", 1)

        page_content = "\n".join(filter(None, content_parts))
        documents.append(Document(page_content=page_content, metadata=metadata_payload))

    logger.info(f"Prepared {len(documents)} hospital documents.")
    return documents

def _prepare_qa_documents():
    documents = []
    
    for qa_index, qa_item in enumerate(data_loader.qa_data):
        content_parts = []
        metadata_payload = {
            "source_doc_id": f"qa_item_{qa_index}",
            "document_type": "qa_data",
            "type": "knowledge_base"
        }
        
        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        context = qa_item.get("context", "")
        source = qa_item.get("source", "")
        
        if question:
            content_parts.append(f"Question: {question}")
            metadata_payload["question"] = question
        
        if answer:
            content_parts.append(f"Answer: {answer}")
            metadata_payload["answer"] = answer[:200]
        
        if context:
            content_parts.append(f"Context: {context}")
            metadata_payload["context"] = context
            metadata_payload["topic"] = context.lower().replace(" ", "_")
        
        if source:
            content_parts.append(f"Source: {source}")
            metadata_payload["source_url"] = source
        
        searchable_content = f"{question} {answer} {context}".strip()
        content_parts.append(f"Searchable Content: {searchable_content}")
        
        key_terms = set()
        for text in [question, answer, context]:
            if text:
                words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
                key_terms.update(words)
        
        metadata_payload["key_terms"] = list(key_terms)[:10]
        
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
        
        if question.lower().startswith(('what is', 'what are', 'what was', 'what were', 'what does', 'what do')):
            metadata_payload["priority"] = 1
        elif question.lower().startswith(('how to', 'how can', 'how do', 'how does', 'how is')):
            metadata_payload["priority"] = 2
        elif question.lower().startswith(('where is', 'where are', 'where can', 'where do', 'where does')):
            metadata_payload["priority"] = 1
        elif question.lower().startswith(('tell me', 'tell', 'provide', 'give', 'give me', 'provide me', 'i need', 'i want', ' i want to')):
            metadata_payload["priority"] = 2
        else:
            metadata_payload["priority"] = 3
        
        tags = []
        if context:
            tags.append(context.lower().replace(" ", "_"))
        tags.append(metadata_payload["category"])
        metadata_payload["tags"] = tags[:3]
        
        summary = f"Q&A about {context if context else 'hospital information'}: {question[:50]}..."
        metadata_payload["summary"] = summary

        page_content = "\n".join(filter(None, content_parts))
        documents.append(Document(page_content=page_content, metadata=metadata_payload))
    
    logger.info(f"Prepared {len(documents)} QA documents.")
    return documents