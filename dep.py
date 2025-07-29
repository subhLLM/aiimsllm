import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIIMSJammuScraper:
    def __init__(self):
        self.base_url = "https://www.aiimsjammu.edu.in"
        self.departments_url = "https://www.aiimsjammu.edu.in/department-here/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.departments_data = []
        
    def get_page(self, url, max_retries=3):
        """Fetch a page with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
    
    def extract_department_links(self):
        """Extract all department links from the main departments page"""
        logger.info("Extracting department links...")
        response = self.get_page(self.departments_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        department_links = []
        
        # Look for department links - adjust selectors based on actual page structure
        # This is a generic approach that looks for links in the main content area
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href')
            if href and ('department' in href.lower() or 'dept' in href.lower() or any(dept in href.lower() for dept in ['cardiology', 'surgery', 'medicine', 'orthopedics', 'pediatrics', 'ent', 'radiology', 'pathology', 'anesthesia', 'dermatology', 'psychiatry', 'neurology', 'oncology', 'ophthalmology', 'urology', 'gynecology', 'emergency', 'microbiology', 'biochemistry', 'pharmacology', 'physiology', 'anatomy', 'forensic', 'community', 'transfusion'])):
                full_url = urljoin(self.base_url, href)
                if full_url not in department_links and self.base_url in full_url:
                    department_links.append(full_url)
        
        # Also look for department links in navigation menus or specific containers
        nav_elements = soup.find_all(['nav', 'div'], class_=re.compile(r'(menu|nav|department)', re.I))
        for nav in nav_elements:
            for link in nav.find_all('a', href=True):
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if (full_url not in department_links and 
                        self.base_url in full_url and 
                        any(dept in full_url.lower() for dept in ['cardiology', 'surgery', 'medicine', 'orthopedics', 'pediatrics', 'ent', 'radiology', 'pathology', 'anesthesia', 'dermatology', 'psychiatry', 'neurology', 'oncology', 'ophthalmology', 'urology', 'gynecology', 'emergency', 'microbiology', 'biochemistry', 'pharmacology', 'physiology', 'anatomy', 'forensic', 'community', 'transfusion'])):
                        department_links.append(full_url)
        
        logger.info(f"Found {len(department_links)} department links")
        return list(set(department_links))  # Remove duplicates
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return None
        return re.sub(r'\s+', ' ', text.strip())
    
    def extract_email(self, text):
        """Extract email from text"""
        if not text:
            return None
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    
    def extract_phone(self, text):
        """Extract phone number from text"""
        if not text:
            return None
        phone_pattern = r'\b(?:\+91[\s-]?)?[789]\d{9}\b'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else None
    
    def extract_faculty_info(self, soup):
        """Extract faculty information from department page"""
        faculty = []
        
        # Look for faculty sections with various possible selectors
        faculty_sections = soup.find_all(['div', 'section'], class_=re.compile(r'(faculty|staff|team|doctor)', re.I))
        
        if not faculty_sections:
            # Try finding by text content
            faculty_sections = soup.find_all(text=re.compile(r'(faculty|staff|team|doctor)', re.I))
            faculty_sections = [elem.parent for elem in faculty_sections if elem.parent]
        
        for section in faculty_sections:
            # Look for individual faculty members
            faculty_items = section.find_all(['div', 'li', 'tr'], class_=re.compile(r'(member|person|doctor|faculty)', re.I))
            
            if not faculty_items:
                # Try to find structured data
                faculty_items = section.find_all(['div', 'p']) + section.find_all('tr')
            
            for item in faculty_items:
                name = None
                designation = None
                email = None
                
                # Try to extract name (usually in bold, h3, h4, or strong tags)
                name_elem = item.find(['strong', 'b', 'h3', 'h4', 'h5'])
                if name_elem:
                    name = self.clean_text(name_elem.get_text())
                
                # If no name found, try looking for "Dr." pattern
                if not name:
                    text_content = item.get_text()
                    dr_match = re.search(r'Dr\.?\s+([A-Za-z\s\.]+)', text_content)
                    if dr_match:
                        name = self.clean_text(dr_match.group(1))
                
                # Extract designation
                designation_patterns = [
                    r'(Professor|Associate Professor|Assistant Professor|Senior Resident|Junior Resident|Consultant|Head|Additional Professor)',
                    r'(Prof\.?|Assoc\.?\s*Prof\.?|Asst\.?\s*Prof\.?)'
                ]
                
                text_content = item.get_text()
                for pattern in designation_patterns:
                    match = re.search(pattern, text_content, re.I)
                    if match:
                        designation = self.clean_text(match.group(0))
                        break
                
                # Extract email
                email = self.extract_email(text_content)
                
                if name and (designation or email):
                    faculty.append({
                        "name": name,
                        "designation": designation,
                        "email": email
                    })
        
        return faculty
    
    def extract_helpdesk_info(self, soup):
        """Extract helpdesk information from department page"""
        helpdesk = {}
        
        # Look for helpdesk/contact sections
        helpdesk_sections = soup.find_all(['div', 'section'], class_=re.compile(r'(helpdesk|contact|phone|emergency)', re.I))
        
        if not helpdesk_sections:
            # Try finding by text content
            helpdesk_text = soup.find_all(text=re.compile(r'(helpdesk|contact|phone|emergency|opd)', re.I))
            helpdesk_sections = [elem.parent for elem in helpdesk_text if elem.parent]
        
        for section in helpdesk_sections:
            text_content = section.get_text()
            
            # Look for OPD information
            opd_phones = re.findall(r'OPD[:\s]*(\d+)', text_content, re.I)
            emergency_phones = re.findall(r'Emergency[:\s]*(\d+)', text_content, re.I)
            
            # Extract phone numbers and try to categorize them
            phone_numbers = re.findall(r'\b[789]\d{9}\b', text_content)
            
            if phone_numbers:
                if 'opd' in text_content.lower():
                    helpdesk['OPD'] = {
                        "Patient Care Manager": phone_numbers[0] if phone_numbers else None,
                        "Screening OPD Helpdesk": phone_numbers[1] if len(phone_numbers) > 1 else None
                    }
                
                if 'emergency' in text_content.lower():
                    helpdesk['Emergency'] = {
                        "Patient Care Manager": phone_numbers[0] if phone_numbers else None,
                        "Nursing Supervisor": phone_numbers[1] if len(phone_numbers) > 1 else None
                    }
        
        return helpdesk if helpdesk else None
    
    def scrape_department(self, dept_url):
        """Scrape individual department page"""
        logger.info(f"Scraping department: {dept_url}")
        response = self.get_page(dept_url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract department name
        department_name = None
        title_elem = soup.find('title')
        if title_elem:
            department_name = self.clean_text(title_elem.get_text())
        
        # Try to get department name from h1 or main heading
        if not department_name:
            h1_elem = soup.find('h1')
            if h1_elem:
                department_name = self.clean_text(h1_elem.get_text())
        
        # Extract from URL if still not found
        if not department_name:
            path = urlparse(dept_url).path
            department_name = path.split('/')[-2] if path.endswith('/') else path.split('/')[-1]
            department_name = department_name.replace('-', ' ').title()
        
        # Extract faculty information
        faculty = self.extract_faculty_info(soup)
        
        # Extract helpdesk information
        helpdesk = self.extract_helpdesk_info(soup)
        
        department_data = {
            "department": department_name,
            "department_website": dept_url,
            "faculty": faculty,
            "helpdesk": helpdesk,
            "last_verified": datetime.now().strftime("%Y-%m-%d")
        }
        
        return department_data
    
    def scrape_all_departments(self):
        """Scrape all departments"""
        logger.info("Starting department scraping...")
        
        # Get all department links
        department_links = self.extract_department_links()
        
        if not department_links:
            logger.warning("No department links found. Trying manual department URLs...")
            # Fallback to common department URLs
            common_departments = [
                'cardiology', 'surgery', 'medicine', 'orthopedics', 'pediatrics', 
                'ent', 'radiology', 'pathology', 'anesthesia', 'dermatology',
                'psychiatry', 'neurology', 'oncology', 'ophthalmology', 'urology',
                'gynecology', 'emergency', 'microbiology', 'biochemistry'
            ]
            department_links = [f"{self.base_url}/{dept}/" for dept in common_departments]
        
        # Scrape each department
        for dept_url in department_links:
            try:
                dept_data = self.scrape_department(dept_url)
                if dept_data:
                    self.departments_data.append(dept_data)
                time.sleep(1)  # Be respectful to the server
            except Exception as e:
                logger.error(f"Error scraping {dept_url}: {e}")
                continue
        
        logger.info(f"Scraped {len(self.departments_data)} departments")
        return self.departments_data
    
    def save_to_json(self, filename="aiims_departments.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.departments_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filename}")

def main():
    scraper = AIIMSJammuScraper()
    
    # Scrape all departments
    departments_data = scraper.scrape_all_departments()
    
    # Save to JSON file
    scraper.save_to_json("aiims_jammu_departments.json")
    
    # Print summary
    print(f"\nScraping completed!")
    print(f"Total departments scraped: {len(departments_data)}")
    print(f"Data saved to: aiims_jammu_departments.json")
    
    # Display sample data
    if departments_data:
        print(f"\nSample department data:")
        print(json.dumps(departments_data[0], indent=2))

if __name__ == "__main__":
    main()