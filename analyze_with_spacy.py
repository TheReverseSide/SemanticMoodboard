import pandas as pd
import spacy

# ---------------------
# LOAD DATA
# ---------------------
df = pd.read_csv("./output/scraped_freedom_sentences.csv")

LANGUAGE_MODELS = {
    "English": {
        "df": df[df["language"] == "English"].copy(),
        "model": "en_core_web_trf"
    }
    # "Spanish": {
    #     "df": df[df["language"] == "Spanish"].copy(),
    #     "model": "es_dep_news_trf"
    # },
    # "Italian": {
    #     "df": df[df["language"] == "Italian"].copy(),
    #     "model": "it_core_news_lg"
    # },
    # "German": {
    #     "df": df[df["language"] == "German"].copy(),
    #     "model": "de_dep_news_trf"
    # },
    # "Swedish": {
    #     "df": df[df["language"] == "Swedish"].copy(),
    #     "model": "sv_core_news_lg"
    # },
}

# ---------------------
# CONFIG
# ---------------------
KEYWORD = "freedom"
results = []

# ---------------------
# ANALYZE SENTENCES (DEPENDENCY ANALYSIS)
# ---------------------
for lang_name, config in LANGUAGE_MODELS.items():
    print(f"Processing: {lang_name}")
    df_lang = config["df"]
    nlp = spacy.load(config["model"])


    for sentence in df_lang["sentence"]:
        doc = nlp(sentence)

        for token in doc:
            if token.text.lower() != KEYWORD:
                continue

            # --- ADJECTIVES modifying "freedom"
            for child in token.children:
                if child.dep_ == "amod":
                    results.append({
                        "keyword": "freedom",
                        "co_word": child.text.lower(),
                        "pos": child.pos_,
                        "dep_type": "adj_modifier",
                        "sentence": sentence
                    })

            # --- VERBS where "freedom" is the subject
            if token.dep_ == "nsubj":
                head = token.head
                if head.pos_ == "VERB":
                    results.append({
                        "keyword": "freedom",
                        "co_word": head.text.lower(),
                        "pos": head.pos_,
                        "dep_type": "subject_of_verb",
                        "sentence": sentence
                    })

            # --- Prepositional phrases like "freedom of X"
            for child in token.children:
                if child.dep_ == "prep":
                    for obj in child.children:
                        if obj.dep_ == "pobj":
                            results.append({
                                "keyword": "freedom",
                                "co_word": obj.text.lower(),
                                "pos": obj.pos_,
                                "dep_type": "prep_object",
                                "sentence": sentence
                            })

  
# ---------------------
# SAVE RESULTS
# ---------------------

co_df = pd.DataFrame(results)
# Removing instances where the keyword (freedom) = the co_word
print(f"before removing dupes {len(co_df)}")
co_df = co_df[co_df["co_word"] != co_df["keyword"]] 
co_df.to_csv("./outputs/spacy_freedom_dependence_analysis.csv", index=False)

print(f"Found {len(co_df)} dependency-linked words.")
print(co_df.head())
