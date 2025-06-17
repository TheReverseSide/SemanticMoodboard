import json
import os
from pathlib import Path

import deepl
import pandas as pd
from dotenv import load_dotenv


# ---------------------
# CONSTANTS & CONFIG
# ---------------------
LANG_CODES = {
    "English": "en",
    "Spanish": "es",
    "Italian": "it",
    "German": "de",
    "Swedish": "sv"
}

CACHE_PATH = Path("cache/translation_cache.json")


# ---------------------
# FUNCTIONS
# ---------------------
def load_translation_cache():
    """Load translation cache from disk or create empty cache."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


def save_translation_cache(cache):
    """Save translation cache to disk."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print("✅ Saved cache.")


def translate_coword_english(row, translator, translation_cache, counter):
    """Translate co_word to English using DeepL API and caching."""
    counter[0] += 1

    co_word = row["co_word"]
    lang_name = row["lang_name"]

    if lang_name == "English":
        print(f"{counter[0]}. Skipping English word: {co_word}")
        return co_word
    
    # Use cache key
    key = f"{lang_name}:{co_word.lower()}"
    if key in translation_cache:
        print(f"{counter[0]}. Using cached translation for: {co_word} ({lang_name}) → {translation_cache[key]}")
        return translation_cache[key]

    try:
        print(f"{counter[0]}. Translating {co_word} from {lang_name}...")
        translated = translator.translate_text(
            co_word,
            source_lang=LANG_CODES[lang_name].upper(),
            target_lang="EN-US"
        )
        translated_word = translated.text.strip().capitalize()

        if translated.detected_source_lang != LANG_CODES[lang_name].upper():
            print(f"Incorrect language detection for: {co_word}. \n" 
                  f"DeepL detected language {translated.detected_source_lang} \n"
                  f"LANG_CODES detected {LANG_CODES[lang_name]}")

        # Catch poor translations (e.g., identical result)
        if not translated_word or translated_word.lower() == co_word.lower():
            print(f"No useful translation for: {co_word}")
            translation_cache[key] = None
            return None

        print(f"Translated to '{translated_word}'")
        translation_cache[key] = translated_word
        return translated_word

    except Exception as e:
        print(f"Error translating '{co_word}': {str(e)}")
        translation_cache[key] = None
        return None


# ---------------------
# MAIN EXECUTION
# ---------------------
def main():
    # Setup
    translation_cache = load_translation_cache()
    load_dotenv()
    api_key = os.getenv("DEEPL_API_KEY")
    translator = deepl.Translator(api_key)
    counter = [0]  # Use list for mutable counter
    
    # Data loading
    df = pd.read_csv("./outputs/spacy_freedom_dependence_analysis.csv")
    
    # Clean up and formatting
    df["co_word"] = df["co_word"].str.strip().str.lower()
    df["pos"] = df["pos"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["lang_name", "co_word", "sentence"])
    df["co_word_and_pos"] = df["co_word"] + "_" + df["pos"]
    
    # Add word counts
    df_freq = (
        df
        .groupby(["lang_name", "co_word"])
        .size()
        .reset_index(name="count")
        .sort_values(["lang_name", "count"], ascending=[True, False])
    )
    
    # Merge and translate
    df_merged = df.merge(df_freq, on=["lang_name", "co_word"], how="left")
    df_merged["english_coword"] = df_merged.apply(
        lambda row: translate_coword_english(row, translator, translation_cache, counter), 
        axis=1
    )
    df_merged["english_coword"] = df_merged["english_coword"].str.strip().str.lower()
    
    # Calculate shared word frequencies
    word_lang_freq = (
        df_merged
        .groupby(["english_coword", "lang_name"])
        .size()
        .reset_index(name="shared_word_frequency")
    )
    df_merged = df_merged.merge(
        word_lang_freq,
        on=["english_coword", "lang_name"],
        how="left"
    )
    
    # Final formatting
    df_merged.columns = [col.strip().lower().replace(" ", "_") for col in df_merged.columns]
    df_merged["combined_label"] = df_merged["english_coword"] + " (" + df_merged["co_word"] + ")"
    
    # Save results
    df_merged.to_csv("./outputs/freedom_viz_ready.csv", index=False)
    df_merged.to_excel("./outputs/freedom_viz_ready_worksheeet.xlsx", index=False, engine='openpyxl')
    save_translation_cache(translation_cache)
    
    print("✅ Cleaned dataset saved as csv and xlsx.")


if __name__ == "__main__":
    main()