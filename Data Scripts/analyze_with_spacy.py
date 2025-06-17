import argparse
from typing import List, Dict, Any

import pandas as pd
import spacy


# ---------------------
# CONSTANTS & CONFIG
# ---------------------
def get_language_models(df):
    """Return language models configuration dict after data is loaded."""
    return {
        "English": {
            "df": df[df["language"] == "English"].copy(),
            "model": "en_core_web_trf",
            "keyword": df[df["language"] == "English"]["source_word"].iloc[0]
        },
        "Spanish": {
            "df": df[df["language"] == "Spanish"].copy(),
            "model": "es_dep_news_trf",
            "keyword": df[df["language"] == "Spanish"]["source_word"].iloc[0]
        },
        "Italian": {
            "df": df[df["language"] == "Italian"].copy(),
            "model": "it_core_news_lg",
            "keyword": df[df["language"] == "Italian"]["source_word"].iloc[0]
        },
        "German": {
            "df": df[df["language"] == "German"].copy(),
            "model": "de_dep_news_trf",
            "keyword": df[df["language"] == "German"]["source_word"].iloc[0]
        }
    }


# ---------------------
# ANALYSIS FUNCTIONS
# ---------------------
def analyze_romance_sentence(sentence: str, lang_name: str, keyword: str, nlp, debug=False) -> list[dict]:
    """Analyze Romance language sentences for dependency relationships with keyword."""
    results = []
    doc = nlp(sentence)
    keyword = keyword.lower()
    keyword_found = False

    # Parts of speech to exclude from results
    junk_pos = {"DET", "PRON", "PART", "CCONJ", "SCONJ", "PUNCT", "ADP"}
    # Dependency relations
    subj_deps = {"nsubj", "nsubjpass"}
    obj_deps = {"obj", "iobj"}
    prep_deps = {"pobj", "obl"}

    for token in doc:
        text_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()
        if lemma_lower != keyword and text_lower != keyword:
            continue
        keyword_found = True

        # 1. Subject-of-verb (including copula)
        if token.dep_ in subj_deps:
            head = token.head
            if head.pos_ in ("AUX", "VERB"):
                # capture the verb
                results.append(_result(token, head, "VERB", sentence, lang_name))
                # capture any adjective/noun complement
                for comp in head.children:
                    if comp.pos_ not in junk_pos:
                        results.append(_result(token, comp, comp.pos_, sentence, lang_name))

        # 2. Object-of-verb
        if token.dep_ in obj_deps:
            head = token.head
            if head.pos_ in ("AUX", "VERB"):
                results.append(_result(token, head, "VERB", sentence, lang_name))

        # 3. Prepositional objects as verbs (e.g., "talk about freedom")
        if token.dep_ in prep_deps:
            prep = token.head
            governor = prep.head
            if governor.pos_ in ("AUX", "VERB"):
                results.append(_result(token, governor, "VERB", sentence, lang_name))

        # 4. Direct modifiers (children of keyword)
        for child in token.children:
            if child.pos_ not in junk_pos:
                results.append(_result(token, child, child.pos_, sentence, lang_name))

        # 5. Coordinated terms
        for other in doc:
            if other.dep_ == "conj" and other.head == token and other.pos_ not in junk_pos:
                results.append(_result(token, other, other.pos_, sentence, lang_name))

    if debug and not results:
        if not keyword_found:
            print(f"X Keyword '{keyword}' not found: {sentence}")
        else:
            print(f"X Keyword '{keyword}' no relations: {sentence}")
    return results


def analyze_german_sentence(sentence: str, lang_name: str, keyword: str, nlp, debug=False) -> list[dict]:
    """Analyze German sentences for dependency relationships with keyword."""
    results = []
    doc = nlp(sentence)
    keyword = keyword.lower()
    keyword_found = False

    junk_pos = {"DET", "PRON", "PART", "CCONJ", "SCONJ", "PUNCT"}
    subj_deps = {"nsubj", "sb"}  # include sb for German model
    obj_deps = {"dobj", "obj", "iobj", "oa", "da", "pobj", "obl"}

    for token in doc:
        text_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()
        if lemma_lower != keyword and text_lower != keyword:
            continue
        keyword_found = True
        head = token.head

        # 1. Subject-of-verb
        if token.dep_ in subj_deps and head.pos_ in ("AUX", "VERB"):
            results.append(_result(token, head, head.pos_, sentence, lang_name))
            for comp in head.children:
                if comp.dep_ in {"acomp", "attr"} and comp.pos_ not in junk_pos:
                    results.append(_result(token, comp, comp.pos_, sentence, lang_name))

        # 2. Object-of-verb
        if token.dep_ in obj_deps and head.pos_ in ("AUX", "VERB"):
            results.append(_result(token, head, head.pos_, sentence, lang_name))

        # 3. Direct modifiers
        for child in token.children:
            if child.pos_ not in junk_pos:
                results.append(_result(token, child, child.pos_, sentence, lang_name))

        # 4. Coordinated terms
        for other in doc:
            if other.dep_ == "conj" and other.head == token and other.pos_ not in junk_pos:
                results.append(_result(token, other, other.pos_, sentence, lang_name))

    if debug and not results:
        if not keyword_found:
            print(f"X Keyword '{keyword}' not found: {sentence}")
        else:
            print(f"X Keyword '{keyword}' no relations: {sentence}")
    return results


def _result(keyword_token, co_word_token, dep_type, sentence, lang_name) -> dict:
    return {
        "keyword": keyword_token.lemma_.lower(),
        "co_word": co_word_token.lemma_.lower(),
        "pos": co_word_token.pos_,
        "dep_type": dep_type,
        "sentence": sentence,
        "lang_name": lang_name
    }


def load_data(test_mode=False):
    """Load sentence data based on test mode setting."""
    if test_mode:
        print("🔍 Running in TEST MODE...")
        return pd.read_csv("./tests/test_scraped_sentences.csv")
    else:
        return pd.read_csv("./outputs/scraped_freedom_sentences.csv")


def analyze_sentences(language_models):
    """Process all sentences in all configured languages."""
    language_analyzers = {
        "German": analyze_german_sentence,
        "English": analyze_romance_sentence,  
        "Spanish": analyze_romance_sentence,
        "Italian": analyze_romance_sentence,
    }
    
    results = []

    for lang_name, config in language_models.items():
        df_target_lang = config["df"]
        print(f"Processing: {lang_name} ({len(df_target_lang)})")
        
        keyword = config["keyword"]
        nlp = spacy.load(config["model"])
        analyze_function = language_analyzers[lang_name]

        for sentence in df_target_lang["sentence"]:
            results.extend(analyze_function(sentence, lang_name, keyword, nlp, debug=True))
    
    return results


def save_results(results):
    """Save analysis results to CSV file."""
    df_dep_analysis_words = pd.DataFrame(results)
    print(f"\n Sentences gathered:\n {df_dep_analysis_words['lang_name'].value_counts()}")
    df_dep_analysis_words.to_csv("./outputs/spacy_freedom_dependence_analysis.csv", index=False)


# ---------------------
# MAIN EXECUTION
# ---------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.test)
    
    # Get language models configuration
    language_models = get_language_models(df)
    
    # Analyze sentences
    results = analyze_sentences(language_models)
    
    # Save results
    save_results(results)


if __name__ == "__main__":
    main()