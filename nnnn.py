import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import json
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIIMSJammuScraper:
    def __init__(self):
        self.base_url = "https://www.aiimsjammu.edu.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directories
        self.out_dir = Path("data/aiims_jammu_complete")
        self.html_dir = self.out_dir / "html_pages"
        self.json_dir = self.out_dir / "structured_data"
        self.media_dir = self.out_dir / "media_links"
        
        for dir_path in [self.html_dir, self.json_dir, self.media_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.visited_urls = set()
        self.all_data = {}
        
    def get_comprehensive_page_list(self):
        """Get comprehensive list of all pages to scrape"""
        return {
            # Main pages
            "home": "/",
            "about": "/about-us/",
            "vision_mission": "/vision-mission/",
            "administration": "/administration/",
            "director_message": "/directors-message/",
            
            # Departments - Medical
            "departments": "/departments/",
            "general_medicine": "/general-medicine/",
            "cardiology": "/cardiology/",
            "neurology": "/neurology/",
            "gastroenterology": "/gastroenterology/",
            "nephrology": "/nephrology/",
            "pulmonology": "/pulmonology/",
            "endocrinology": "/endocrinology/",
            "rheumatology": "/rheumatology/",
            "dermatology": "/dermatology/",
            "psychiatry": "/psychiatry/",
            
            # Departments - Surgical
            "general_surgery": "/general-surgery/",
            "cardiothoracic_surgery": "/cardiothoracic-surgery/",
            "neurosurgery": "/neurosurgery/",
            "plastic_surgery": "/plastic-surgery/",
            "orthopaedics": "/orthopaedics/",
            "urology": "/urology/",
            "ophthalmology": "/ophthalmology/",
            "ent": "/ent/",
            "obstetrics_gynaecology": "/obstetrics-and-gynaecology/",
            
            # Departments - Diagnostic
            "radiodiagnosis": "/radiodiagnosis/",
            "pathology": "/pathology/",
            "microbiology": "/microbiology/",
            "biochemistry": "/biochemistry/",
            "pharmacology": "/pharmacology/",
            "physiology": "/physiology/",
            "anatomy": "/anatomy/",
            "forensic_medicine": "/forensic-medicine/",
            "community_medicine": "/community-medicine/",
            
            # Departments - Support
            "anaesthesiology": "/anaesthesiology/",
            "emergency_medicine": "/emergency-medicine/",
            "nuclear_medicine": "/nuclear-medicine/",
            "radiation_oncology": "/radiation-oncology/",
            "transfusion_medicine": "/transfusion-medicine/",
            "hospital_administration": "/hospital-administration/",
            
            # Specialties
            "paediatrics": "/paediatrics/",
            "paediatric_surgery": "/paediatric-surgery/",
            "neonatology": "/neonatology/",
            "paediatric_cardiology": "/paediatric-cardiology/",
            
            # Nursing and Allied Sciences
            "nursing": "/nursing/",
            "physiotherapy": "/physiotherapy/",
            "medical_social_service": "/medical-social-service/",
            "clinical_psychology": "/clinical-psychology/",
            "dietetics": "/dietetics/",
            "medical_records": "/medical-records/",
            
            # Academic sections
            "academics": "/academics/",
            "mbbs": "/mbbs/",
            "md_ms": "/md-ms/",
            "dm_mch": "/dm-mch/",
            "nursing_courses": "/nursing-courses/",
            "paramedical_courses": "/paramedical-courses/",
            "phd": "/phd/",
            "admissions": "/admissions/",
            "examination": "/examination/",
            "library": "/library/",
            
            # Research
            "research": "/research/",
            "research_projects": "/research-projects/",
            "publications": "/publications/",
            "conferences": "/conferences/",
            "workshops": "/workshops/",
            "cmr": "/cmr/",
            "institutional_ethics_committee": "/institutional-ethics-committee/",
            
            # Patient care
            "patient_care": "/patient-care/",
            "opd_services": "/opd-services/",
            "emergency_services": "/emergency-services/",
            "icu_services": "/icu-services/",
            "operation_theatres": "/operation-theatres/",
            "diagnostic_services": "/diagnostic-services/",
            "pharmacy": "/pharmacy/",
            "ambulance_services": "/ambulance-services/",
            
            # Information sections
            "notices": "/notices/",
            "news": "/news/",
            "events": "/events/",
            "media": "/media/",
            "press_release": "/press-release/",
            "photo_gallery": "/photo-gallery/",
            "video_gallery": "/video-gallery/",
            
            # Administrative
            "tenders": "/tenders/",
            "recruitment": "/recruitment/",
            "career": "/career/",
            "rti": "/rti/",
            "annual_report": "/annual-report/",
            "financial_information": "/financial-information/",
            "citizen_charter": "/citizen-charter/",
            
            # Contact and support
            "contact_us": "/contact-us/",
            "directory": "/directory/",
            "phone_directory": "/phone-directory/",
            "important_links": "/important-links/",
            "feedback": "/feedback/",
            "grievance": "/grievance/",
            "hospital_map": "/hospital-map/",
            "location": "/location/",
            
            # COVID-19 related
            "covid_info": "/archives/category/covid-19/",
            "covid_guidelines": "/covid-guidelines/",
            "covid_vaccination": "/covid-vaccination/",
            
            # Archives and categories
            "archives": "/archives/",
            "category_announcements": "/category/announcements/",
            "category_news": "/category/news/",
            "category_events": "/category/events/",
            "category_tenders": "/category/tenders/",
            "category_recruitment": "/category/recruitment/",
            
            # Faculty and staff
            "faculty": "/faculty/",
            "staff": "/staff/",
            "visiting_faculty": "/visiting-faculty/",
            "emeritus_faculty": "/emeritus-faculty/",
            
            # Quality and accreditation
            "nabh": "/nabh/",
            "nabl": "/nabl/",
            "quality_policy": "/quality-policy/",
            "patient_safety": "/patient-safety/",
            "infection_control": "/infection-control/",
            
            # Downloads
            "downloads": "/downloads/",
            "forms": "/forms/",
            "application_forms": "/application-forms/",
            "hospital_forms": "/hospital-forms/",
            
            # Miscellaneous
            "sitemap": "/sitemap/",
            "disclaimer": "/disclaimer/",
            "privacy_policy": "/privacy-policy/",
            "terms_conditions": "/terms-conditions/",
            "accessibility": "/accessibility/",
            "screen_reader": "/screen-reader/",
        }
    
    def discover_additional_links(self, soup):
        """Discover additional links from the current page"""
        links = set()
        
        # Find all internal links
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(self.base_url, href)
            
            # Only include internal links
            if self.base_url in full_url and full_url not in self.visited_urls:
                parsed_url = urlparse(full_url)
                clean_path = parsed_url.path.rstrip('/')
                if clean_path:
                    links.add(clean_path)
        
        return links
    
    def extract_structured_data(self, soup, page_name):
        """Extract structured data from the page"""
        data = {
            'page_name': page_name,
            'timestamp': datetime.now().isoformat(),
            'title': '',
            'meta_description': '',
            'headings': [],
            'links': [],
            'images': [],
            'tables': [],
            'lists': [],
            'contact_info': [],
            'dates': [],
            'downloads': []
        }
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            data['title'] = title_tag.get_text().strip()
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            data['meta_description'] = meta_desc.get('content', '')
        
        # Headings
        for i in range(1, 7):
            headings = soup.find_all(f'h{i}')
            for heading in headings:
                data['headings'].append({
                    'level': i,
                    'text': heading.get_text().strip()
                })
        
        # Links
        for link in soup.find_all('a', href=True):
            data['links'].append({
                'text': link.get_text().strip(),
                'href': link['href']
            })
        
        # Images
        for img in soup.find_all('img'):
            data['images'].append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        # Tables
        for table in soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text().strip() for cell in cells]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                data['tables'].append(table_data)
        
        # Lists
        for ul in soup.find_all(['ul', 'ol']):
            list_items = [li.get_text().strip() for li in ul.find_all('li')]
            if list_items:
                data['lists'].append(list_items)
        
        # Contact information (emails, phones)
        text_content = soup.get_text()
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text_content)
        
        # Phone patterns
        phone_pattern = r'(\+91|0)?[\s-]?[6-9]\d{9}|\d{3,4}[\s-]?\d{6,8}'
        phones = re.findall(phone_pattern, text_content)
        
        data['contact_info'] = {
            'emails': list(set(emails)),
            'phones': list(set(phones))
        }
        
        # Date patterns
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}\b'
        dates = re.findall(date_pattern, text_content, re.IGNORECASE)
        data['dates'] = list(set(dates))
        
        # Download links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(ext in href.lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']):
                data['downloads'].append({
                    'text': link.get_text().strip(),
                    'href': href
                })
        
        return data
    
    def crawl_page(self, name, url_path):
        """Crawl a single page and extract data"""
        full_url = urljoin(self.base_url, url_path)
        
        if full_url in self.visited_urls:
            return None
        
        logger.info(f"📥 Fetching: {full_url}")
        
        try:
            response = self.session.get(full_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Clean and extract visible text
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            text = soup.get_text(separator="\n")
            
            # Save HTML text
            html_file = self.html_dir / f"{name}.txt"
            html_file.write_text(text.strip(), encoding="utf-8")
            
            # Extract structured data
            structured_data = self.extract_structured_data(soup, name)
            
            # Save structured data
            json_file = self.json_dir / f"{name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            # Discover additional links
            new_links = self.discover_additional_links(soup)
            
            self.visited_urls.add(full_url)
            self.all_data[name] = structured_data
            
            logger.info(f"✅ Saved: {name} ({len(new_links)} new links discovered)")
            
            return new_links
            
        except Exception as e:
            logger.error(f"❌ Error fetching {full_url}: {e}")
            return None
    
    def run_comprehensive_scrape(self):
        """Run comprehensive scraping of AIIMS Jammu website"""
        logger.info("🚀 Starting comprehensive AIIMS Jammu scraping...")
        
        # Get initial page list
        pages = self.get_comprehensive_page_list()
        additional_links = set()
        
        # Scrape all predefined pages
        for name, url_path in pages.items():
            new_links = self.crawl_page(name, url_path)
            if new_links:
                additional_links.update(new_links)
            
            # Be respectful - add delay between requests
            time.sleep(1)
        
        # Scrape discovered links
        logger.info(f"📋 Found {len(additional_links)} additional links to scrape")
        
        for i, link_path in enumerate(additional_links):
            if len(self.visited_urls) > 500:  # Limit to prevent infinite crawling
                break
                
            link_name = f"discovered_{i+1}"
            self.crawl_page(link_name, link_path)
            time.sleep(1)
        
        # Save summary
        summary = {
            'total_pages_scraped': len(self.visited_urls),
            'scraping_date': datetime.now().isoformat(),
            'pages_scraped': list(self.all_data.keys()),
            'base_url': self.base_url
        }
        
        summary_file = self.out_dir / "scraping_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Scraping completed! Total pages: {len(self.visited_urls)}")
        logger.info(f"📁 Data saved in: {self.out_dir}")
        
        return summary

def main():
    """Main function to run the scraper"""
    scraper = AIIMSJammuScraper()
    
    try:
        summary = scraper.run_comprehensive_scrape()
        print(f"\n{'='*50}")
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Total pages scraped: {summary['total_pages_scraped']}")
        print(f"Data saved in: {scraper.out_dir}")
        print(f"Check scraper.log for detailed logs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        print(f"❌ Scraping failed: {e}")

if __name__ == "__main__":
    main()