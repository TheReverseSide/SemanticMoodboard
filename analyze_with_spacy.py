import pandas as pd
import spacy

# ---------------------
# LOAD DATA
# ---------------------
df = pd.read_csv("./outputs/scraped_freedom_sentences.csv")

LANGUAGE_MODELS = {
    "English": {
        "df": df[df["language"] == "English"].copy(),
        "model": "en_core_web_trf",
        "keyword": df[df["language"] == "English"]["source_word"].iloc[0]
    },
    # "Spanish": {
    #     "df": df[df["language"] == "Spanish"].copy(),
    #     "model": "es_dep_news_trf",
    #     "keyword": df[df["language"] == "Spanish"]["source_word"].iloc[0]
    # },
    # "Italian": {
    #     "df": df[df["language"] == "Italian"].copy(),
    #     "model": "it_core_news_lg",
    #     "keyword": df[df["language"] == "Italian"]["source_word"].iloc[0]
    # },
    # "German": {
    #     "df": df[df["language"] == "German"].copy(),
    #     "model": "de_dep_news_trf",
    #     "keyword": df[df["language"] == "German"]["source_word"].iloc[0]
    # },
    # "Swedish": {
    #     "df": df[df["language"] == "Swedish"].copy(),
    #     "model": "sv_core_news_lg"
    # },
}

# ---------------------
# SET UP ANALYZERS
# ---------------------
def analyze_german_sentence(sentence: str, lang_name: str, keyword: str, nlp) -> list[dict]:
    """
    Analyzes a German sentence for linguistic relationships with the keyword.
    Accounts for German-specific dependency tags like 'nk' (noun modifiers) and 'sb' (subject).
    Returns a list of dictionaries containing co-occurring words tied to the keyword.
    """
    doc = nlp(sentence)
    keyword = keyword.lower()
    german_results = []

    for token in doc:
        token_text = token.text.lower()

        # Case 1: the token IS the keyword itself
        if token_text == keyword:
            # Inspect children for modifiers or prepositions
            for child in token.children:
                # German-specific: 'nk' = noun kernel modifier (often adjective or determiner)
                if child.dep_ in ("nk", "mo") and child.pos_ == "ADJ":
                    german_results.append({
                        "keyword": keyword,
                        "co_word": child.text.lower(),
                        "pos": child.pos_,
                        "dep_type": "adj_modifier",
                        "sentence": sentence,
                        "lang_name": lang_name
                    })

                # Prepositional modifier structure (if it exists)
                if child.dep_ in ("mo", "mnr", "adv"):
                    for obj in child.children:
                        if obj.dep_ in ("nk", "oa"):  # 'oa' = accusative object (can include nouns)
                            german_results.append({
                                "keyword": keyword,
                                "co_word": obj.text.lower(),
                                "pos": obj.pos_,
                                "dep_type": "prep_object",
                                "sentence": sentence,
                                "lang_name": lang_name
                            })

        # Case 2: token is a dependent where its HEAD is the keyword
        elif token.head.text.lower() == keyword:
            # Again check for noun modifier
            if token.dep_ in ("nk", "mo") and token.pos_ == "ADJ":
                german_results.append({
                    "keyword": keyword,
                    "co_word": token.text.lower(),
                    "pos": token.pos_,
                    "dep_type": "adj_modifier",
                    "sentence": sentence,
                    "lang_name": lang_name
                })

            # If this token is a preposition, its child might be the object
            if token.dep_ in ("mo", "mnr", "adv"):
                for obj in token.children:
                    if obj.dep_ in ("nk", "oa"):
                        german_results.append({
                            "keyword": keyword,
                            "co_word": obj.text.lower(),
                            "pos": obj.pos_,
                            "dep_type": "prep_object",
                            "sentence": sentence,
                            "lang_name": lang_name
                        })

        # Case 3: the keyword is the subject of a verb (German label = 'sb')
        if token.dep_ == "sb" and token_text == keyword:
            head = token.head
            if head.pos_ == "VERB":
                german_results.append({
                    "keyword": keyword,
                    "co_word": head.text.lower(),
                    "pos": head.pos_,
                    "dep_type": "subject_of_verb",
                    "sentence": sentence,
                    "lang_name": lang_name
                })

    return german_results

def analyze_german_sentence(sentence: str, lang_name: str, keyword: str, nlp) -> list[dict]:
    doc = nlp(sentence)
    keyword = keyword.lower()
    german_results = []

    for token in doc:
        token_text = token.text.lower()

        # --- Direct Match: token is the keyword
        if token_text == keyword:
            for child in token.children:
                if child.dep_ in ("nk", "mo") and child.pos_ == "ADJ":
                    german_results.append({
                        "keyword": keyword,
                        "co_word": child.lemma_.lower(),
                        "pos": child.pos_,
                        "dep_type": "adj_modifier",
                        "sentence": sentence,
                        "lang_name": lang_name
                    })

                if child.dep_ in ("poss", "compound", "det"):
                    german_results.append({
                        "keyword": keyword,
                        "co_word": child.lemma_.lower(),
                        "pos": child.pos_,
                        "dep_type": "noun_modifier",
                        "sentence": sentence,
                        "lang_name": lang_name
                    })

                if child.dep_ in ("mo", "mnr", "adv"):
                    for obj in child.children:
                        if obj.dep_ in ("nk", "oa"):
                            german_results.append({
                                "keyword": keyword,
                                "co_word": obj.lemma_.lower(),
                                "pos": obj.pos_,
                                "dep_type": "prep_object",
                                "sentence": sentence,
                                "lang_name": lang_name
                            })

        # --- Keyword is subject of a verb ('sb' = subject)
        if token.dep_ == "sb" and token_text == keyword:
            head = token.head
            if head.pos_ == "VERB":
                german_results.append({
                    "keyword": keyword,
                    "co_word": head.lemma_.lower(),
                    "pos": head.pos_,
                    "dep_type": "subject_of_verb",
                    "sentence": sentence,
                    "lang_name": lang_name
                })

        # --- Keyword is object of a verb ('oa', 'da' = accusative/dative object)
        if token_text == keyword and token.dep_ in ("oa", "da"):
            head = token.head
            if head.pos_ == "VERB":
                german_results.append({
                    "keyword": keyword,
                    "co_word": head.lemma_.lower(),
                    "pos": head.pos_,
                    "dep_type": "object_of_verb",
                    "sentence": sentence,
                    "lang_name": lang_name
                })

        # --- Prepositional object structure (if keyword is the 'pobj' equivalent)
        if token_text == keyword and token.dep_ == "nk" and token.head.dep_ in ("mo", "mnr"):
            prep = token.head
            governor = prep.head
            german_results.append({
                "keyword": keyword,
                "co_word": prep.lemma_.lower(),
                "pos": prep.pos_,
                "dep_type": "prep_linked",
                "sentence": sentence,
                "lang_name": lang_name
            })

        # --- Keyword is the HEAD of a dependent
        elif token.head.text.lower() == keyword:
            if token.dep_ in ("nk", "mo") and token.pos_ == "ADJ":
                german_results.append({
                    "keyword": keyword,
                    "co_word": token.lemma_.lower(),
                    "pos": token.pos_,
                    "dep_type": "adj_modifier",
                    "sentence": sentence,
                    "lang_name": lang_name
                })

            if token.dep_ in ("mo", "mnr", "adv"):
                for obj in token.children:
                    if obj.dep_ in ("nk", "oa"):
                        german_results.append({
                            "keyword": keyword,
                            "co_word": obj.lemma_.lower(),
                            "pos": obj.pos_,
                            "dep_type": "prep_object",
                            "sentence": sentence,
                            "lang_name": lang_name
                        })

    return german_results


LANGUAGE_ANALYZERS = {
    "German": analyze_german_sentence,
    "English": analyze_romance_sentence,
    "Spanish": analyze_romance_sentence, # todo - create own methods
    "Italian": analyze_romance_sentence, # todo - create own methods
}

# ---------------------
# ANALYZE SENTENCES (DEPENDENCY ANALYSIS)
# ---------------------
results = []

for lang_name, config in LANGUAGE_MODELS.items():
    print(f"Processing: {lang_name}")
    df_target_lang = config["df"]
    print(f"DF:  {len(df_target_lang)}")
    keyword = config["keyword"]
    nlp = spacy.load(config["model"])

    analyze_function = LANGUAGE_ANALYZERS[lang_name]

    for sentence in df_target_lang["sentence"]:
        results.extend(analyze_function(sentence, lang_name, keyword, nlp))
  
# ---------------------
# SAVE RESULTS
# ---------------------
df_dep_analysis_words = pd.DataFrame(results)

print(f"\n Sentences gathered: {df_dep_analysis_words['lang_name'].value_counts()}")
df_dep_analysis_words.to_csv("./outputs/spacy_freedom_dependence_analysis.csv", index=False)
