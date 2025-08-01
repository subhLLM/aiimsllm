import json
import logging
import os
from config import HOSPITAL_MODEL_JSON_PATH, QA_PAIRS_JSON_PATH, ALLOWED_FILES
from collections import Counter

logger = logging.getLogger(__name__)

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
        
        if self.hospital_data:
            rooms, departments, doctors, services = set(), set(), set(), set()
            lifts, stairs, exits, entrances, ramps = set(), set(), set(), set(), set()
            hospitals, buildings, venues = set(), set(), set()

            for item in self.hospital_data:
                if item.get("hospitalName"):
                    hospitals.add(item["hospitalName"])

                loc_ctx = item.get("locationContext", {})
                if loc_ctx.get("venueName"):
                    venues.add(loc_ctx["venueName"])
                if loc_ctx.get("buildingName"):
                    buildings.add(loc_ctx["buildingName"])

                room_details = item.get("roomDetails", {})
                if room_details.get("roomName"):
                    rooms.add(room_details["roomName"])
                if room_details.get("roomNumber"):
                    rooms.add(str(room_details["roomNumber"]))

                for dept in self.ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        departments.add(dept["departmentName"])
                    
                    for service_name in self.ensure_list(dept.get("relatedServices", [])):
                        services.add(service_name)
                    
                    for doctor in self.ensure_list(dept.get("doctors", [])):
                        if doctor.get("name"):
                            doctors.add(doctor["name"])

                for service_item in self.ensure_list(item.get("servicesOffered", [])):
                    if service_item.get("serviceName"):
                        services.add(service_item["serviceName"])

                access = item.get("accessibility", {}).get("nearestAccessPoints", {})
                for group, collector in [("lifts", lifts), ("stairs", stairs), ("emergencyExits", exits),
                                        ("entrances", entrances), ("ramps", ramps)]:
                    for ap in self.ensure_list(access.get(group, [])):
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
        
        if self.qa_data:
            topics = set()
            sources = set()
            
            for qa_item in self.qa_data:
                context = qa_item.get("context", "")
                if context:
                    topics.add(context)
                
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
        
        if self.hospital_data:
            for item in self.hospital_data:
                if item.get("hospitalType"):
                    tags.add(item["hospitalType"].lower().replace(" ", "_"))
                
                room_details = item.get("roomDetails", {})
                if room_details.get("roomType"):
                    tags.add(room_details["roomType"].lower().replace(" ", "_"))
                if room_details.get("roomSubType"):
                    tags.add(room_details["roomSubType"].lower().replace(" ", "_"))
                
                for dept in self.ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        tags.add(dept["departmentName"].lower().replace(" ", "_"))
                
                for service in self.ensure_list(item.get("servicesOffered", [])):
                    if service.get("serviceName"):
                        tags.add(service["serviceName"].lower().replace(" ", "_"))
                
                if item.get("emergencyServices"):
                    tags.add("emergency_services")
                
                if item.get("accessibility", {}).get("isWheelchairAccessible"):
                    tags.add("wheelchair_accessible")
        
        if self.qa_data:
            for qa_item in self.qa_data:
                context = qa_item.get("context", "")
                if context:
                    context_tag = context.lower().replace(" ", "_")
                    tags.add(context_tag)
        
        return sorted(tags)

    def get_metadata_tag_counts(self):
        counter = Counter()
        
        if self.hospital_data:
            for item in self.hospital_data:
                if item.get("hospitalType"):
                    counter[item["hospitalType"].lower().replace(" ", "_")] += 1
                
                room_details = item.get("roomDetails", {})
                if room_details.get("roomType"):
                    counter[room_details["roomType"].lower().replace(" ", "_")] += 1
                
                for dept in self.ensure_list(item.get("departments", [])):
                    if dept.get("departmentName"):
                        counter[dept["departmentName"].lower().replace(" ", "_")] += 1
                
                for service in self.ensure_list(item.get("servicesOffered", [])):
                    if service.get("serviceName"):
                        counter[service["serviceName"].lower().replace(" ", "_")] += 1
        
        if self.qa_data:
            for qa_item in self.qa_data:
                context = qa_item.get("context", "")
                if context:
                    context_tag = context.lower().replace(" ", "_")
                    counter[context_tag] += 1
        
        return dict(counter.most_common())

    @staticmethod
    def ensure_list(val):
        if isinstance(val, list):
            return val
        elif val:
            return [val]
        return []