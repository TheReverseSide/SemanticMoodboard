
# Usage
1. selenium_scraper.py -- Goes and grabs sentences from Tatoeba
    - Removes any near-duplicates/duplicate sentences
    - outputs 'scraped_freedom_sentences.csv'
2. analyze_with_spacy.py -- Creates a dependency context csv using linguistic analysis
    - outputs 'spacy_freedom_dependence_analysis.csv'
3. Sentiment Analysis - takes that data from spacy and appends a hugging face sentiment analysis
    - outputs 'freedom_dependence_sentiment.csv'
4. Cleans and prepares data for export to Tableau
    - outputs 'freedom_tableau_ready.csv'
5. Load into Tableau and visualize

# Venv usage that I always forget
cd into scripts folder -> Activate.ps1 (powershell)

# Misc
- pip freeze > requirements.txt
- Swedish didnt have alot of entries in tatoeba so I omitted it
- Sentence length should probably be adjusted per language, as a small refinement