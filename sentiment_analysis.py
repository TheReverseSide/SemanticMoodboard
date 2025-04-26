from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load dependency output
df = pd.read_csv("./outputs/spacy_freedom_dependence_analysis.csv")

# Deduplicate on sentence to avoid reprocessing
unique_sentences = df["sentence"].drop_duplicates().tolist()

sentiment_scores = {}
for sentence in unique_sentences:
    try:
        result = sentiment_pipeline(sentence)[0]
        sentiment_scores[sentence] = result["label"]
    except Exception as e:
        print(f"Error on: {sentence[:30]}... → {e}")
        sentiment_scores[sentence] = "UNKNOWN"

df["sentiment"] = df["sentence"].map(sentiment_scores)

# OPTIONAL: map "1 star" → "negative", "3 star" → "neutral", etc.
def map_stars(label):
    if "1" in label or "2" in label:
        return "NEGATIVE"
    elif "3" in label:
        return "NEUTRAL"
    elif "4" in label or "5" in label:
        return "POSITIVE"
    return "UNKNOWN"

df["sentiment_simple"] = df["sentiment"].apply(map_stars)

df.to_csv("./outputs/freedom_dependence_sentiment.csv", index=False)
print("✅ Exported CSV with sentiment.")
