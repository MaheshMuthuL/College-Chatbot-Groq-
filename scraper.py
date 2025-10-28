import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from collections import deque

# Configuration
BASE_URL = "https://saveetha.ac.in/"  # CHANGE THIS to your college website
MAX_PAGES = 100  # Maximum number of pages to scrape
DELAY = 1  # Delay between requests (seconds) - be respectful!
OUTPUT_DIR = "scraped_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CollegeWebScraper:
    def __init__(self, base_url, max_pages=100, delay=1):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.urls_to_visit = deque([base_url])
        
        # Headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def is_valid_url(self, url):
        """
        Check if URL is valid and belongs to the same domain.
        """
        parsed = urlparse(url)
        
        # Check if it's the same domain
        if parsed.netloc != self.domain:
            return False
        
        # Skip non-HTML files
        excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        return True
    
    def get_page_content(self, url):
        """
        Fetch and parse a web page.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, html_content):
        """
        Extract clean text from HTML content.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, html_content, current_url):
        """
        Extract all valid links from the page.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        links = []
        
        for link in soup.find_all('a', href=True):
            url = urljoin(current_url, link['href'])
            
            # Remove fragments and query parameters for deduplication
            url = url.split('#')[0].split('?')[0]
            
            if self.is_valid_url(url) and url not in self.visited_urls:
                links.append(url)
        
        return links
    
    def save_to_file(self, url, content):
        """
        Save scraped content to a text file.
        """
        # Create a safe filename from URL
        filename = urlparse(url).path.strip('/').replace('/', '_')
        if not filename:
            filename = 'homepage'
        filename = f"{filename}.txt"
        
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write("="*80 + "\n\n")
                f.write(content)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    
    def scrape(self):
        """
        Main scraping function.
        """
        print(f"Starting web scraping from {self.base_url}")
        print(f"Maximum pages to scrape: {self.max_pages}")
        print("="*80)
        
        pages_scraped = 0
        
        while self.urls_to_visit and pages_scraped < self.max_pages:
            url = self.urls_to_visit.popleft()
            
            if url in self.visited_urls:
                continue
            
            print(f"\n[{pages_scraped + 1}/{self.max_pages}] Scraping: {url}")
            
            # Mark as visited
            self.visited_urls.add(url)
            
            # Get page content
            html_content = self.get_page_content(url)
            if not html_content:
                continue
            
            # Extract text
            text_content = self.extract_text(html_content)
            
            # Save to file
            if text_content and len(text_content.strip()) > 100:  # Only save if substantial content
                self.save_to_file(url, text_content)
                pages_scraped += 1
            
            # Extract and queue new links
            new_links = self.extract_links(html_content, url)
            for link in new_links:
                if link not in self.visited_urls:
                    self.urls_to_visit.append(link)
            
            # Be respectful - add delay
            time.sleep(self.delay)
        
        print("\n" + "="*80)
        print(f"Scraping completed! Total pages scraped: {pages_scraped}")
        print(f"Files saved in: {OUTPUT_DIR}/")

# Alternative: Scrape specific pages from a list
def scrape_specific_pages(urls_list):
    """
    Scrape specific pages from a predefined list of URLs.
    """
    scraper = CollegeWebScraper(BASE_URL)
    
    print(f"Scraping {len(urls_list)} specific pages...")
    
    for idx, url in enumerate(urls_list, 1):
        print(f"\n[{idx}/{len(urls_list)}] Scraping: {url}")
        
        html_content = scraper.get_page_content(url)
        if html_content:
            text_content = scraper.extract_text(html_content)
            if text_content:
                scraper.save_to_file(url, text_content)
        
        time.sleep(scraper.delay)
    
    print(f"\nCompleted! Files saved in: {OUTPUT_DIR}/")

# Main execution
if __name__ == "__main__":
    # Option 1: Automatic crawling (recommended)
    scraper = CollegeWebScraper(
        base_url=BASE_URL,
        max_pages=MAX_PAGES,
        delay=DELAY
    )
    scraper.scrape()
    
    # Option 2: Scrape specific pages (uncomment to use)
    # specific_urls = [
    #     "https://your-college.edu/admissions",
    #     "https://your-college.edu/academics",
    #     "https://your-college.edu/about",
    #     "https://your-college.edu/contact",
    # ]
    # scrape_specific_pages(specific_urls)
