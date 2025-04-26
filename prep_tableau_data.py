import pandas as pd

# Load annotated dataset
df = pd.read_csv("./outputs/freedom_dependence_sentiment.csv")

# Optional: Remove any exact duplicates
df = df.drop_duplicates()

word_counts = (
    df.groupby(["co_word", "sentiment_simple"])
    .size()
    .reset_index(name="word_sentiment_count")
)

df = df.merge(word_counts, on=["co_word", "sentiment_simple"], how="left")

# Normalize column names just in case (Tableau is picky)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Create helper column for Tableau filters
df["co_word_and_pos"] = df["co_word"] + " (" + df["pos"] + ")"

# Save the cleaned version
df.to_csv("./outputs/freedom_tableau_ready.csv", index=False)
print("✅ Cleaned dataset saved for Tableau.")
