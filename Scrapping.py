!apt-get update
!apt-get install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
!pip install selenium
!pip install pillow
from google.colab import drive
drive.mount('/content/drive')

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse

# Path to save
save_dir = "/content/drive/MyDrive/scraped_images_narender"
os.makedirs(save_dir, exist_ok=True)

# Selenium setup
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

def clean_url(url):
    return url.split("?")[0]

def scroll_and_collect(driver, num_needed, collected_urls):
    while len(collected_urls) < num_needed:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            srcset = img.get_attribute("srcset")
            if srcset:
                highest_res = srcset.strip().split(",")[-1].split()[0]
                cleaned = clean_url(highest_res)
                collected_urls.add(cleaned)
        print(f"Collected: {len(collected_urls)} URLs")
    return collected_urls

def download_images(urls, save_dir, batch_num, resize_dim=(256, 256)):
    count = 0
    for i, url in enumerate(urls):
        try:
            img_data = requests.get(url, timeout=10).content
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img = img.resize(resize_dim)
            img.save(os.path.join(save_dir, f"batch{batch_num}___{i}.jpg"))
            count += 1
        except:
            pass
    print(f"Downloaded {count} images in batch {batch_num}")

driver.get("https://unsplash.com")
total_images = 44000
batch_size = 2000
batch_size = 2000

for batch in range(1, (total_images // batch_size) + 1):
    print(f"Starting batch {batch}")
    image_urls = set()
    image_urls = scroll_and_collect(driver, batch_size, image_urls)
    download_images(list(image_urls), save_dir, batch_num=batch)
    time.sleep(5)

print("Total images saved:", len(os.listdir(save_dir)))
