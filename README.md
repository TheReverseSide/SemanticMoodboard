
# Readme

## Purpose

This project examines the differences and nuances in the usage of the word 'freedom' between the languages English, German, Italian and Spanish.

As an American, I’ve noticed how often we use “freedom” in vague, broad, and sometimes contradictory ways. My European friends often tease me about it—so I thought: let’s actually look at how “freedom” is used in different languages. Where do definitions across languages agree, and where do they collide?

### Approach

I wanted to scale gradually—starting with clean data and controlled analysis before diving into messier, real-world language. I gathered sentences from native speakers, focusing on the five languages I understand: English, German, Spanish, Italian, and Swedish.

I started with Tatoeba.org, a database of sentences translated by native speakers. Tatoeba has limitations—short, context-less sentences—but the advantage is availability of data. Plus, it supports exact keyword search. I wrote a small script using Selenium to automatically gather sentences across five languages, with throttling and respectful delays.

After gathering the data, we can then analyze it and hopefully find some insights

My assumptions for English, freedom would sound:

- Militaristic verbs: fight for freedom, defend our freedom, sacrifices for freedom
- Rights discourse: free speech, second amendment
- Demands: more freedom, retain existing freedom
- Emotionally intense sentiment, especially around loss or struggle

### Process & Methods

1. Scraped Tatoeba for translated sentences of “freedom,” filtering by minimum sentence length.
2. Deduplicated semantically using sentence-transformers and cosine similarity to avoid near-duplicate noise.
3. Performed dependency parsing with SpaCy, extracting relationships: subjects, objects, modifiers, prepositional links, and coordinated terms.
4. Translated co-words into English via DeepL, caching results for performance.
5. Exported a unified CSV for analysis and visualization.

## Scripts

1. selenium_scraper.py
    - Goes and grabs sentences from Tatoeba
    - Removes any near-duplicates/duplicate sentences
    - outputs 'scraped_freedom_sentences.csv'
2. analyze_with_spacy.py
    - Creates a dependency context csv using linguistic analysis
    - outputs 'spacy_freedom_dependence_analysis.csv'
3. Sentiment Analysis
    - takes that data from spacy and appends a hugging face sentiment analysis
    - NOT IN USE
4. prep_viz_data.py
    - Cleans and prepares data for export to Tableau
    - outputs 'freedom_tableau_ready.csv'
5. dash_app.py
    - Creates visualizations in a Dash webapp
