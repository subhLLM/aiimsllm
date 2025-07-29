import json
from datetime import datetime
import uuid

def load_json_file(filename):
    """Load JSON data from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {filename}")
        return []

def merge_hospital_data(locations_data, doctors_data):
    """Merge location and doctor data into hospital building format"""
    merged_data = []
    
    # Create a dictionary for quick doctor lookup by locationId
    doctors_by_location = {}
    for doctor in doctors_data:
        location_id = doctor.get('locationId')
        if location_id:
            if location_id not in doctors_by_location:
                doctors_by_location[location_id] = []
            doctors_by_location[location_id].append(doctor)
    
    # Process each location
    for location in locations_data:
        poly_id = location.get('properties', {}).get('polyId')
        if not poly_id:
            continue
            
        # Find matching doctors for this location
        matching_doctors = doctors_by_location.get(poly_id, [])
        
        # Get primary doctor (first one) for main entity
        primary_doctor = matching_doctors[0] if matching_doctors else None
        
        # Create merged record
        merged_record = {
            "id": location.get('_id', str(uuid.uuid4())),
            "locationId": poly_id,
            "nodeId": location.get('properties', {}).get('nodeId', ''),
            "locationContext": {
                "campusId": "campus-uuid",
                "hospitalName": "AIIMS Jammu",
                "hospitalType": "General",
                "buildingName": "Diagnostic",
                "tower": "A",
                "floor": location.get('floor', 1),
                "zone": "North Wing",
                "indoor": True,
                "areaType": "Academic Campus"
            },
            "physical": {
                "name": location.get('name', ''),
                "type": "Room",
                "subType": location.get('element', {}).get('subType', 'office'),
                "structure": {
                    "capacity": "1",
                    "areaSqFt": 200,
                    "shape": "Rectangular",
                    "flooringType": "Tiled"
                },
                "coordinates": {
                    "cartesian": {
                        "x": location.get('coordinateX', 0),
                        "y": location.get('coordinateY', 0),
                        "center": {
                            "x": location.get('centerX', 0),
                            "y": location.get('centerY', 0)
                        },
                        "door": {
                            "x": location.get('doorX', 0),
                            "y": location.get('doorY', 0)
                        }
                    },
                    "geo": {
                        "latitude": float(location.get('properties', {}).get('latitude', 0)),
                        "longitude": float(location.get('properties', {}).get('longitude', 0))
                    }
                },
                "door": {
                    "type": "Single",
                    "mechanism": "Manual",
                    "motion": "Push-Pull",
                    "smartLock": False
                },
                "utilities": {
                    "powerBackup": True,
                    "lighting": "LED",
                    "temperatureControl": "Central AC"
                }
            },
            "functional": {
                "purpose": "Doctor's Office" if primary_doctor else "General Room",
                "associatedEntity": {},
                "accessLevel": "Staff Only",
                "availability": {
                    "daysOpen": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "startTime": "09:00",
                    "endTime": "17:00"
                },
                "booking": {
                    "enabled": True,
                    "method": "Online Portal",
                    "url": f"https://booking.aimmsjammu.com/room/{location.get('name', '')}",
                    "approvalRequired": True,
                    "notes": "Visitor pass required at entry gate. Booking needs admin approval.",
                    "cancellationPolicy": "Bookings can be cancelled 24 hours prior."
                },
                "responseChannels": [
                    {
                        "type": "Helpdesk",
                        "description": "Contact the building helpdesk for immediate assistance.",
                        "contact": {
                            "phone": "098-26548765",
                            "email": "helpdesk@aimmsjammu.in"
                        },
                        "operatingHours": {
                            "mondayToFriday": {"start": "09:00", "end": "17:00"}
                        }
                    },
                    {
                        "type": "Security",
                        "description": "Contact security for access issues or emergencies.",
                        "contact": {"phone": "078-26548700"}
                    }
                ]
            },
            "accessibility": {
                "isWheelchairAccessible": True,
                "features": [
                    "Elevator Access",
                    "Braille Signage",
                    "Stairs Access",
                    "Wide Entrance",
                    "Assistive Lighting"
                ],
                "nearestAccessPoints": {
                    "lifts": location.get('lifts', []),
                    "stairs": location.get('stairs', []),
                    "emergencyExits": [],
                    "ramps": location.get('ramps', []),
                    "entrances": location.get('entries', [])
                },
                "amenities": ["WiFi", "Drinking Water", "Restroom Nearby", "Fire Extinguisher"]
            },
            "media": {
                "images": [
                    {
                        "url": f"https://example.com/images/room_{location.get('name', '')}_1.jpg",
                        "altText": f"Interior view of Room {location.get('name', '')}",
                        "type": "interior"
                    }
                ]
            },
            "status": {
                "operational": True,
                "underMaintenance": False,
                "lastInspected": datetime.now().strftime("%Y-%m-%d"),
                "compliance": ["Fire Safety", "Accessibility Act 2021"]
            },
            "metadata": {
                "tags": [],
                "summary": "",
                "entity_names": [],
                "lastUpdated": datetime.now().isoformat() + "Z",
                "createdBy": "admin@aiimsjammu.edu.in"
            }
        }
        
        # Add doctor information if available
        if primary_doctor:
            # Extract working hours
            working_days = primary_doctor.get('workingDays', [])
            operating_hours = {}
            
            for day_info in working_days:
                day = day_info.get('day', '').lower()
                if day == 'th.f':
                    day = 'thursday'
                
                if day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                    operating_hours['mondayToFriday'] = {
                        "start": day_info.get('openingTime', '09:00'),
                        "end": day_info.get('closingTime', '17:00')
                    }
                    break
            
            # Create services list based on speciality
            services = []
            speciality = primary_doctor.get('speciality', '').lower()
            if 'anaesthesia' in speciality:
                services = [
                    "Pain Management",
                    "Anesthesia Administration", 
                    "ICU Sedation Monitoring",
                    "Surgical Consultation"
                ]
            else:
                services = ["General Consultation", "Medical Examination"]
            
            merged_record["functional"]["associatedEntity"] = {
                "entityType": "Doctor",
                "id": primary_doctor.get('_id', ''),
                "name": primary_doctor.get('name', ''),
                "about": primary_doctor.get('about', ''),
                "specialization": primary_doctor.get('speciality', ''),
                "industry": primary_doctor.get('speciality', ''),
                "departmentName": primary_doctor.get('unitName', '').lower(),
                "contact": {
                    "email": f"{primary_doctor.get('name', '').lower().replace(' ', '').replace('dr', '')}@aimmsjammu.com",
                    "phone": "+918587096914",
                    "website": primary_doctor.get('profile', '')
                },
                "operatingHours": operating_hours,
                "servicesOffered": services
            }
            
            # Update availability based on doctor's schedule
            if working_days:
                days_open = []
                for day_info in working_days:
                    day = day_info.get('day', '')
                    if day == 'Th.F':
                        days_open.extend(['Thursday', 'Friday'])
                    elif day:
                        days_open.append(day.capitalize())
                
                merged_record["functional"]["availability"]["daysOpen"] = list(set(days_open))
                
                if working_days:
                    merged_record["functional"]["availability"]["startTime"] = working_days[0].get('openingTime', '09:00')
                    merged_record["functional"]["availability"]["endTime"] = working_days[0].get('closingTime', '17:00')
            
            # Generate tags and metadata
            doctor_name_parts = primary_doctor.get('name', '').lower().split()
            tags = [
                primary_doctor.get('name', '').lower(),
                primary_doctor.get('speciality', '').lower(),
                "doctor office",
                "consultation",
                "healthcare",
                "opd",
                "wheelchair accessible",
                f"room {location.get('name', '')}",
                "academic campus",
                "aiims jammu"
            ]
            
            entity_names = [
                primary_doctor.get('name', '').lower(),
                primary_doctor.get('speciality', '').lower(),
                f"room {location.get('name', '')}",
                "doctor office",
                "aiims jammu",
                "north wing",
                "diagnostic",
                "tower a"
            ]
            
            merged_record["metadata"]["tags"] = tags
            merged_record["metadata"]["entity_names"] = entity_names
            merged_record["metadata"]["summary"] = f"{primary_doctor.get('name', '')} ({primary_doctor.get('speciality', '')}) consults in Room {location.get('name', '')} on Floor {location.get('floor', 1)} of the Academic Building at AIIMS Jammu. The room is wheelchair accessible and available for outpatient services during OPD hours."
        
        merged_data.append(merged_record)
    
    return merged_data

def save_merged_data(data, filename):
    """Save merged data to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Merged data saved to {filename}")
        print(f"Total records: {len(data)}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def main():
    """Main function to execute the merging process"""
    print("Hospital Data Merger Script")
    print("=" * 40)
    
    # Load input files
    print("Loading input files...")
    locations_data = load_json_file('a.json')
    doctors_data = load_json_file('b.json')
    
    if not locations_data or not doctors_data:
        print("Error: Could not load input files")
        return
    
    print(f"Loaded {len(locations_data)} locations and {len(doctors_data)} doctors")
    
    # Merge data
    print("Merging data...")
    merged_data = merge_hospital_data(locations_data, doctors_data)
    
    # Save merged data
    output_filename = 'merged_hospital_building.json'
    if save_merged_data(merged_data, output_filename):
        print(f"Successfully created {output_filename}")
        
        # Print summary
        print("\nSummary:")
        for record in merged_data:
            doctor_name = record.get('functional', {}).get('associatedEntity', {}).get('name', 'No doctor assigned')
            room_name = record.get('physical', {}).get('name', 'Unknown')
            print(f"  Room {room_name}: {doctor_name}")
    else:
        print("Failed to save merged data")

if __name__ == "__main__":
    main()