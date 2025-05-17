import argparse
import numpy as np
import pandas as pd
import time
import unicodedata
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from sentence_transformers import SentenceTransformer, util

# ------------------------
# CONFIG
# ------------------------
TARGETS = [
    {"language": "English", "word": "freedom", "lang_code": "eng", "html_lang": "en"},
    {"language": "German", "word": "Freiheit", "lang_code": "deu", "html_lang": "de"},
    {"language": "Spanish", "word": "libertad", "lang_code": "spa", "html_lang": "es"},
    {"language": "Italian", "word": "libertà", "lang_code": "ita", "html_lang": "it"},
    # {"language": "Swedish", "word": "frihet", "lang_code": "swe", "html_lang": "sv"}
]

MAX_PAGES = 40
DELAY = 2  # seconds between page loads
BASE_URL = "https://tatoeba.org/en/sentences/search"
OUTPUT_PATH = "./outputs/scraped_freedom_sentences.csv"

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

# Set up CLI test flag
parser = argparse.ArgumentParser(description="Scrape Tatoeba sentences.")
parser.add_argument('--test', action='store_true', help='Run in test mode with limited scraping')
args = parser.parse_args()

if args.test:
    print("Running in TEST mode: Only scraping one English language page")
    TARGETS = [TARGETS[0]]
    MAX_PAGES = 5

# Run the scraper for each language
for target in TARGETS:
    print(f"\nScraping {target['language']}...")

    for page in range(1, MAX_PAGES + 1):
        query = f"{BASE_URL}?from={target['lang_code']}&query={target['word']}&page={page}&word_count_min=4"
        print(f"Searching at {query}")
        driver.get(query)
        time.sleep(DELAY)

        try:
            sentence_divs = driver.find_elements(By.CSS_SELECTOR, "div.text")

            for div in sentence_divs:
                # only keep sentences written in the target language
                lang = div.get_attribute("lang")
                if lang == target["html_lang"]:
                    sentence = div.text.strip()
                    print(f"→ {sentence}")
                    if sentence:
                        all_sentences.append({
                            "language": target["language"],
                            "source_word": target["word"],
                            "sentence": sentence
                        })

            print(f"Page {page}: Found {len(sentence_divs)} sentence elements. \n")
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue

def normalize(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    return text

'''Take sentences and remove near-duplicate sentences'''
def deduplicate_embeddings(sentences: list, threshold=0.95) -> list[str]:
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device='cuda')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    
    keep = []
    dropped = set()
    n = len(sentences)

    for i in range(n):
        if i in dropped:
            continue
        keep.append(i)
        for j in range(i + 1, n):
            if cosine_scores[i][j] >= threshold:
                dropped.add(j)

    print("\nDuplicates:")
    for i in dropped:
        print(sentences[i])
    
    return [sentences[i] for i in keep]

# ------------------------
# CLEANUP & SAVE
# ------------------------
driver.quit()

df = pd.DataFrame(all_sentences)

# Normalize the DF sentences
df["sentence"] = df["sentence"].apply(normalize)

# Remove exact duplicates
print(f"\nLength before dropping duplicates: {df.shape[0]}")
print(f"Exact duplicates to be dropped: {df[df.duplicated(subset='sentence', keep='first')]}")
df = df.drop_duplicates(subset="sentence")
print(f"Length after dropping duplicates: {df.shape[0]}\n")

# Remove highly similar sentences with sentence transformer
sentences = df["sentence"].tolist()
print(f"Sentences: {sentences} \n")
print(f"Length before removing duplicates: {len(sentences)}")
unique_sentences = deduplicate_embeddings(sentences)
print(f"Length after removing duplicates: {len(unique_sentences)} \n")

deduped_df = df[df["sentence"].isin(unique_sentences)].copy() # EXACT MATCH, so case sensitivity

deduped_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\nScraping complete. Saved {len(deduped_df)} sentences to {OUTPUT_PATH}")
