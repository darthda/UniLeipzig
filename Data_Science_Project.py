import praw
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import pipeline
import time

# Benötigte libraries für Data Cleaning

import re
import string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize import ByteLevelBPETokenizer
from nltk.corpus import stopwords
import spacy
import nltk
from symspellpy import SymSpell, Verbosity
from langdetect import detect

# Reddit-API einrichten
reddit = praw.Reddit(
    client_id="KQZHY2dGalJMF3zo20Ihkg",
    client_secret="y4u1OAupdHZhC2N-N4urFgVX2zcmCA",
    user_agent="US_Wahl_Analyse"
)

# Zero-Shot-Klassifikationsmodell laden
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Funktion: Abrufen von Posts aus einem Subreddit mit Paginierung
def get_reddit_posts(subreddit_name, keyword, max_posts=5000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    last_post = None

    while len(posts) < max_posts:
        for submission in subreddit.search(keyword, limit=1000, params={'after': last_post}):
            posts.append({
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "url": submission.url,
                "num_comments": submission.num_comments,
                "created": submission.created,
                "author": str(submission.author)
            })
            last_post = submission.id  # Speichert den ID des letzten Posts für die nächste Abfrage

            if len(posts) >= max_posts:
                break
        time.sleep(1)  # Pause, um API-Limits zu vermeiden

    df = pd.DataFrame(posts)
    return df

# Funktion: Zero-Shot-Klassifikation anwenden
def classify_post(text, labels):
    try:
        result = classifier(text, labels)
        return result['labels'][0], result['scores'][0]
    except Exception as e:
        print(f"Fehler bei der Klassifikation: {e}")
        return None, 0.0

# Subreddit-Liste und Einstellungen
subreddit_list = ["politics", "PoliticalDiscussion", "Conservative", "democrats", "republican"] ## Eventuell Liste von Subreddits erweitern
keyword = "election"
max_posts = 5000  # Maximale Anzahl an Posts pro Subreddit

# Abrufen der Posts aus mehreren Subreddits
all_posts = []
for subreddit in subreddit_list:
    print(f"Abrufen von Posts aus r/{subreddit}...")
    try:
        df = get_reddit_posts(subreddit, keyword, max_posts=max_posts)
        all_posts.append(df)
    except Exception as e:
        print(f"Fehler beim Abrufen von r/{subreddit}: {e}")
df_all_posts = pd.concat(all_posts, ignore_index=True)

# Zeitstempel in ein lesbares Datum umwandeln
df_all_posts['created_date'] = pd.to_datetime(df_all_posts['created'], unit='s')

# Zeitraum: Kamalas Nominierung bis Wahltag (2024)
start_date = datetime(2024, 7, 31)
end_date = datetime(2024, 11, 5)

# Filtere die Posts nach Zeitraum
filtered_posts = df_all_posts[(df_all_posts['created_date'] >= start_date) &
                               (df_all_posts['created_date'] <= end_date)]

# Data Cleaning

# Sicherstellen, dass alle benötigten Resourcen geladen sind
nltk.download('punkt')
nltk.download('stopwords')

# Laden der Spacy-Pipeline
nlp = spacy.load("en_core_web_sm")


# 1. Funktion: Kleinschreibung

def to_lowercase(text):
    return text.lower()


# 2. Funktion: Entfernen von HTML-Tags

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# 3. Funktion: Entfernen von Stopwörtern

def remove_stopwords(text):
    stops = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stops])


# 4. Funktion: Entfernen von Sonderzeichen

def remove_special_characters(text):
    # Erlaubt: Buchstaben, Zahlen, Leerzeichen und gängige Satzzeichen
    return re.sub(r"[^a-zA-Z0-9.,!?\-\' ]", "", text)


# 5. Funktion: Tokenisierung

def tokenize_text(text):
    word_tokens = word_tokenize(text)
    byte_level_tokenizer = ByteLevelBPETokenizer()
    return word_tokens, byte_level_tokenizer.encode(text).tokens


# 6. Funktion: Lemmatisierung und Filtern nach POS-Tags

def lemmatize_and_filter(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}]
    return filtered_tokens


# 7. Funktion: Entfernen von nicht-ASCII-Zeichen

def remove_non_ascii(text):
    return text.encode("ascii", errors="ignore").decode()


# 8. Funktion: Expansion von Kontraktionen

def expand_contractions(text):
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "I'm": "I am",
        "you're": "you are",
        "it's": "it is",
        "she's": "she is",
        "he's": "he is",
        "we're": "we are",
        "they're": "they are",
        "I've": "I have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "I'd": "I would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would",
        "there's": "there is",
        "here's": "here is",
        "what's": "what is",
        "that's": "that is",
        "let's": "let us",
        "who's": "who is",
        "where's": "where is",
        "how's": "how is",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mightn't": "might not",
        "mustn't": "must not"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

# Extra: remove numbers

def remove_numbers(text):
    return re.sub(r"\b\d+\b", "", text)

# Extra: Normalisierung von Wörtern mit Buchstabenwiederholungen

def normalize_repeated_characters(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

# Extra: Entfernen von sehr kurzen Wörtern

def remove_short_words(text, min_length=2):
    return " ".join([word for word in text.split() if len(word) >= min_length])

# Filterung nicht englischer Inhalte

def filter_non_english_text(text):
    try:
        if detect(text) == "en":
            return text
        else:
            return ""
    except:
        return ""

# 9. Funktion: Rechtschreibkorrektur

def correct_spelling(text):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

    corrected_words = []
    for word in word_tokenize(text):
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


# 10. Funktion: Entfernen von übermäßiger Interpunktion

def remove_excessive_punctuation(text):
    return re.sub(r"[!?&]{2,}", "!", re.sub(r"[\.]{2,}", ".", text))


# Komplettpipeline für einen Text

def preprocess_text(text):
    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = expand_contractions(text)
    text = remove_numbers(text)  # NEU: Entfernen von Zahlen
    text = normalize_repeated_characters(text)  # NEU: Normalisierung von Wiederholungen
    text = remove_short_words(text)  # NEU: Entfernen von kurzen Wörtern
    text = filter_non_english_text(text)  # NEU: Entfernen von nicht-englischen Texten
    text = remove_stopwords(text)
    text = remove_special_characters(text)
    text = remove_non_ascii(text)
    text = correct_spelling(text)
    text = remove_excessive_punctuation(text)
    tokens, byte_tokens = tokenize_text(text)
    filtered_tokens = lemmatize_and_filter(text)
    return {
        "cleaned_text": text,
        "tokens": tokens,
        "byte_tokens": byte_tokens,
        "filtered_tokens": filtered_tokens
    }

# Kai damit kannst du es auf einem kleinen Beispielpost testen evtl. zum Debuggen!!!

# Beispielanwendung auf einen Reddit-Post
# example_post = "<p>This is an example post!!! It has HTML tags, contractions like can't, and excessive punctuation!!!</p>"
# processed = preprocess_text(example_post)
# print(processed)

# Kategorien für Zero-Shot-Klassifikation
labels = ["pro-Trump", "pro-Harris", "neutral"]

# Zero-Shot-Klassifikation auf Titel und Inhalte anwenden
filtered_posts['classification'], filtered_posts['confidence'] = zip(*filtered_posts['title'].apply(
    lambda x: classify_post(x, labels)))

# Klassifikationen zählen
classification_counts = filtered_posts['classification'].value_counts()

# Visualisierung: Verteilung der Klassifikationen
plt.figure(figsize=(10, 6))
classification_counts.plot(kind='bar')
plt.title("Verteilung der Klassifikationen (pro-Trump, pro-Harris, neutral)")
plt.xlabel("Kategorie")
plt.ylabel("Anzahl der Posts")
plt.xticks(rotation=0)
plt.show()

# Visualisierung: Erwähnungen über die Zeit
filtered_posts['pro_trump'] = filtered_posts['classification'] == "pro-Trump"
filtered_posts['pro_harris'] = filtered_posts['classification'] == "pro-Harris"
mentions_over_time = filtered_posts.groupby('created_date').agg({
    'pro_trump': 'sum',
    'pro_harris': 'sum'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(mentions_over_time['created_date'], mentions_over_time['pro_trump'], label='pro-Trump')
plt.plot(mentions_over_time['created_date'], mentions_over_time['pro_harris'], label='pro-Harris')
plt.title("Erwähnungen von pro-Trump und pro-Harris über die Zeit (2024)")
plt.xlabel("Datum")
plt.ylabel("Anzahl der Posts")
plt.legend()
plt.show()

# Ergebnisse speichern
filtered_posts.to_csv("classified_reddit_posts_2024.csv", index=False)

# Ergebnisse ausgeben
print("Klassifikationsübersicht:")
print(classification_counts)