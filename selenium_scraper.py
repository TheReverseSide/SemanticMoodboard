import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ------------------------
# CONFIG
# ------------------------

TARGETS = [
    {"language": "English", "word": "freedom", "lang_code": "eng", "html_lang": "en"},
    # {"language": "German", "word": "Freiheit", "lang_code": "deu", "html_lang": "de"},
    # {"language": "Spanish", "word": "libertad", "lang_code": "spa", "html_lang": "es"},
    # {"language": "Italian", "word": "libertà", "lang_code": "ita", "html_lang": "it"},
    # {"language": "Swedish", "word": "frihet", "lang_code": "swe", "html_lang": "sv"}
]

MAX_PAGES = 20
DELAY = 2  # seconds between page loads
BASE_URL = "https://tatoeba.org/en/sentences/search"
OUTPUT_CSV = "./output/scraped_freedom_sentences.csv"

# ------------------------
# SELENIUM SETUP
# ------------------------

chrome_options = Options()
chrome_options.add_argument("--headless")  # Run browser invisibly
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

# ------------------------
# SCRAPING LOGIC
# ------------------------

all_sentences = []

for target in TARGETS:
    print(f"\nScraping {target['language']}...")

    for page in range(1, MAX_PAGES + 1):
        query = f"{BASE_URL}?from={target['lang_code']}&query={target['word']}&page={page}&word_count_min=6"
        print(f"Searching at {query}")
        driver.get(query)
        time.sleep(DELAY)

        try:
            sentence_divs = driver.find_elements(By.CSS_SELECTOR, "div.text")

            for div in sentence_divs:
                # only keep sentences written in the target language
                lang = div.get_attribute("lang")
                if lang == target["html_lang"]:
                    # span = div.find_element(By.CLASS_NAME, "sentence")
                    sentence = div.text.strip()
                    print(f"→ {sentence}")
                    if sentence:
                        all_sentences.append({
                        "language": target["language"],
                        "source_word": target["word"],
                        "sentence": sentence
                    })


            print(f"Page {page}: Found {len(sentence_divs)} sentence elements.")
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue

# ------------------------
# CLEANUP & SAVE
# ------------------------

driver.quit()

df = pd.DataFrame(all_sentences)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nScraping complete. Saved {len(df)} sentences to {OUTPUT_CSV}")
