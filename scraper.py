import requests
import pandas as pd
import time
import random
from bs4 import BeautifulSoup

def scrape_zameen_page(page_number, city="lahore"):
    """
    Example scraper for Zameen.com (Note: Zameen has Cloudflare protection. 
    In a real scenario, you might need Selenium, Playwright, or API endpoints).
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    url = f"https://www.zameen.com/Homes/{city}-{page_number}.html"
    
    print(f"Scraping {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch page. Status: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all('li', role='article')
        
        data = []
        for li in listings:
            try:
                # Extract basic info (The selectors here are illustrative and depend on Zameen's current DOM)
                price_elem = li.find(attrs={"aria-label": "Price"})
                price = price_elem.text.strip() if price_elem else None
                
                loc_elem = li.find(attrs={"aria-label": "Location"})
                location = loc_elem.text.strip() if loc_elem else None
                
                area_elem = li.find(attrs={"aria-label": "Area"})
                area = area_elem.text.strip() if area_elem else None
                
                beds_elem = li.find(attrs={"aria-label": "Beds"})
                beds = beds_elem.text.strip() if beds_elem else "0"
                
                baths_elem = li.find(attrs={"aria-label": "Baths"})
                baths = baths_elem.text.strip() if baths_elem else "0"
                
                if price and location:
                    data.append({
                        "price_str": price,
                        "location": location,
                        "city": city,
                        "area_str": area,
                        "bedrooms": beds,
                        "baths": baths,
                        "property_type": "House" # Defaulting for this URL
                    })
            except Exception as e:
                continue
                
        return data
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def main():
    all_data = []
    cities = ['lahore', 'karachi', 'islamabad']
    
    for city in cities:
        for page in range(1, 6): # Scrape first 5 pages as demo
            page_data = scrape_zameen_page(page, city)
            all_data.extend(page_data)
            
            # Polite rate limiting
            time.sleep(random.uniform(2, 5))
            
    df = pd.DataFrame(all_data)
    df.to_csv("data/scraped_data_raw.csv", index=False)
    print(f"Scraped {len(df)} listings. Saved to data/scraped_data_raw.csv")

if __name__ == "__main__":
    main()
