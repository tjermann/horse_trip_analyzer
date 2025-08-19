import os
import time
import requests
from typing import Optional, Dict, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import yt_dlp
from loguru import logger
from pathlib import Path


class TJKVideoScraper:
    def __init__(self, download_dir: str = "data/videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.setup_driver()
        
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def get_video_info(self, race_code: str) -> Dict[str, Any]:
        url = f"https://www.tjk.org/EN/YarisSever/Info/YarisVideoKosu/Kosu?KosuKodu={race_code}"
        
        try:
            logger.info(f"Fetching race info for code: {race_code}")
            self.driver.get(url)
            
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            
            time.sleep(3)
            
            video_element = self.driver.find_element(By.TAG_NAME, "video")
            video_url = video_element.get_attribute("src")
            
            if not video_url:
                source_elements = video_element.find_elements(By.TAG_NAME, "source")
                if source_elements:
                    video_url = source_elements[0].get_attribute("src")
            
            race_info = self.extract_race_metadata()
            
            return {
                "race_code": race_code,
                "video_url": video_url,
                "page_url": url,
                "metadata": race_info
            }
            
        except Exception as e:
            logger.error(f"Error fetching video info: {e}")
            return None
    
    def extract_race_metadata(self) -> Dict[str, Any]:
        metadata = {}
        
        try:
            info_elements = self.driver.find_elements(By.CSS_SELECTOR, ".race-info, .kosu-bilgi")
            for elem in info_elements:
                text = elem.text
                if ":" in text:
                    key, value = text.split(":", 1)
                    metadata[key.strip()] = value.strip()
            
            horse_table = self.driver.find_elements(By.CSS_SELECTOR, "table.horses, table.at-tablosu")
            if horse_table:
                horses = []
                rows = horse_table[0].find_elements(By.TAG_NAME, "tr")[1:]
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 3:
                        horse_info = {
                            "number": cols[0].text.strip(),
                            "name": cols[1].text.strip(),
                            "jockey": cols[2].text.strip() if len(cols) > 2 else ""
                        }
                        horses.append(horse_info)
                metadata["horses"] = horses
                
        except Exception as e:
            logger.warning(f"Could not extract all metadata: {e}")
            
        return metadata
    
    def download_video(self, video_url: str, race_code: str) -> Optional[str]:
        output_path = self.download_dir / f"race_{race_code}.mp4"
        
        if output_path.exists():
            logger.info(f"Video already exists: {output_path}")
            return str(output_path)
        
        ydl_opts = {
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
            'format': 'best[ext=mp4]/best',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video for race {race_code}")
                ydl.download([video_url])
            logger.success(f"Downloaded video to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            
            try:
                logger.info("Attempting direct download with requests...")
                response = requests.get(video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logger.success(f"Downloaded video to: {output_path}")
                return str(output_path)
            except Exception as e2:
                logger.error(f"Direct download also failed: {e2}")
                return None
    
    def scrape_race(self, race_code: str) -> Optional[str]:
        video_info = self.get_video_info(race_code)
        
        if not video_info or not video_info.get("video_url"):
            logger.error(f"Could not find video URL for race {race_code}")
            return None
        
        video_path = self.download_video(video_info["video_url"], race_code)
        
        if video_path:
            metadata_path = Path(video_path).with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(video_info, f, indent=2)
            logger.info(f"Saved metadata to: {metadata_path}")
        
        return video_path
    
    def close(self):
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def __del__(self):
        self.close()


if __name__ == "__main__":
    scraper = TJKVideoScraper(download_dir="data/videos")
    
    try:
        video_path = scraper.scrape_race("194367")
        if video_path:
            print(f"Successfully downloaded race video to: {video_path}")
    finally:
        scraper.close()