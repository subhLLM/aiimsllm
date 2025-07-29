import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIIMSJammuScraper:
    def __init__(self, base_url="https://www.aiimsjammu.edu.in/", max_pages=100):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls = set()
        self.scraped_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def is_valid_url(self, url):
        """Check if URL belongs to AIIMS Jammu domain"""
        parsed = urlparse(url)
        return parsed.netloc == urlparse(self.base_url).netloc
    
    def clean_text(self, text):
        """Clean and normalize text content"""
        if not text:
            return ""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', ' ', text)
        return text
    
    def extract_links(self, soup, current_url):
        """Extract all valid internal links from the page"""
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(current_url, href)
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.add(full_url)
        return links
    
    def generate_qa_pairs(self, content, title, url):
        """Generate question-answer pairs from content"""
        qa_pairs = []
        
        # Main page information
        if title and content:
            qa_pairs.append({
                "question": f"What is {title}?",
                "answer": content[:500] + "..." if len(content) > 500 else content,
                "context": title,
                "source": url
            })
        
        # Split content into sections for more specific Q&A
        paragraphs = content.split('\n\n') if content else []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 100:  # Only process substantial paragraphs
                # Extract potential topics from headings or first sentence
                sentences = paragraph.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 20:
                        # Create Q&A based on content
                        question = self.generate_question_from_content(first_sentence, title)
                        qa_pairs.append({
                            "question": question,
                            "answer": paragraph.strip(),
                            "context": f"{title} - Section {i+1}",
                            "source": url
                        })
        
        return qa_pairs
    
    def generate_question_from_content(self, content, context):
        """Generate appropriate questions based on content"""
        content_lower = content.lower()
        
        # Common question patterns based on content type
        if any(word in content_lower for word in ['department', 'faculty', 'staff']):
            return f"What about the departments and faculty at {context}?"
        elif any(word in content_lower for word in ['admission', 'course', 'program']):
            return f"What are the admission procedures and courses at {context}?"
        elif any(word in content_lower for word in ['research', 'study', 'project']):
            return f"What research activities are conducted at {context}?"
        elif any(word in content_lower for word in ['facility', 'infrastructure', 'equipment']):
            return f"What facilities are available at {context}?"
        elif any(word in content_lower for word in ['contact', 'address', 'phone']):
            return f"How to contact {context}?"
        elif any(word in content_lower for word in ['history', 'established', 'founded']):
            return f"What is the history of {context}?"
        else:
            return f"Tell me about {context}"
    
    def scrape_page(self, url):
        """Scrape content from a single page"""
        try:
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = self.clean_text(soup.title.string)
            elif soup.find('h1'):
                title = self.clean_text(soup.find('h1').get_text())
            
            # Extract main content
            content_areas = []
            
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('div', class_=re.compile(r'content|main', re.I))
            if main_content:
                content_areas.append(main_content)
            else:
                # Fallback to common content containers
                for selector in ['article', '.content', '.main-content', '#content', '#main']:
                    elements = soup.select(selector)
                    content_areas.extend(elements)
            
            # If no specific content area found, use body
            if not content_areas:
                content_areas = [soup.find('body')]
            
            # Extract text content
            all_text = []
            for area in content_areas:
                if area:
                    # Extract headings and paragraphs
                    for element in area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
                        text = self.clean_text(element.get_text())
                        if text and len(text) > 20:  # Only include substantial text
                            all_text.append(text)
            
            content = '\n\n'.join(all_text)
            
            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(content, title, url)
            self.scraped_data.extend(qa_pairs)
            
            # Extract links for further crawling
            links = self.extract_links(soup, url)
            
            return links
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return set()
    
    def crawl_website(self):
        """Main crawling function using BFS approach"""
        logger.info("Starting website crawl...")
        
        urls_to_visit = deque([self.base_url])
        pages_scraped = 0
        
        while urls_to_visit and pages_scraped < self.max_pages:
            current_url = urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            # Scrape the current page
            new_links = self.scrape_page(current_url)
            pages_scraped += 1
            
            # Add new links to the queue
            for link in new_links:
                if link not in self.visited_urls:
                    urls_to_visit.append(link)
            
            # Be respectful - add delay between requests
            time.sleep(1)
            
            logger.info(f"Progress: {pages_scraped}/{self.max_pages} pages scraped")
        
        logger.info(f"Crawling completed. Scraped {pages_scraped} pages, generated {len(self.scraped_data)} Q&A pairs")
    
    def save_to_json(self, filename="aiims_jammu_data.json"):
        """Save scraped data to JSON file"""
        try:
            # Remove duplicates based on question similarity
            unique_data = []
            seen_questions = set()
            
            for item in self.scraped_data:
                question_key = item['question'].lower().strip()
                if question_key not in seen_questions:
                    seen_questions.add(question_key)
                    unique_data.append(item)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data saved to {filename}. Total unique Q&A pairs: {len(unique_data)}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None

def main():
    """Main execution function"""
    print("AIIMS Jammu Website Scraper for RAG Chatbot")
    print("=" * 50)
    
    # Initialize scraper
    scraper = AIIMSJammuScraper(max_pages=50)  # Adjust max_pages as needed
    
    try:
        # Start crawling
        scraper.crawl_website()
        
        # Save data
        filename = scraper.save_to_json()
        
        if filename:
            print(f"\n✅ Scraping completed successfully!")
            print(f"📁 Data saved to: {filename}")
            print(f"📊 Total Q&A pairs generated: {len(scraper.scraped_data)}")
            print(f"🔗 URLs visited: {len(scraper.visited_urls)}")
            
            # Show sample data
            if scraper.scraped_data:
                print(f"\n📝 Sample Q&A pair:")
                sample = scraper.scraped_data[0]
                print(f"Question: {sample['question']}")
                print(f"Answer: {sample['answer'][:200]}...")
                print(f"Context: {sample['context']}")
                print(f"Source: {sample['source']}")
        else:
            print("❌ Failed to save data")
            
    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user")
        if scraper.scraped_data:
            filename = scraper.save_to_json("aiims_jammu_data_partial.json")
            print(f"💾 Partial data saved to: {filename}")
    
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()