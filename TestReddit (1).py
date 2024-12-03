import praw
import pandas as pd
import spacy
from datetime import datetime
import matplotlib.pyplot as plt
from textblob import TextBlob

# Reddit-API einrichten
reddit = praw.Reddit(
    client_id="KQZHY2dGalJMF3zo20Ihkg",
    client_secret="y4u1OAupdHZhC2N-N4urFgVX2zcmCA",
    user_agent="US_Wahl_Analyse"
)

# Funktion: Abrufen von Posts aus einem Subreddit
def get_reddit_posts(subreddit_name, keyword, limit=1090):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.search(keyword, limit=limit):
        posts.append({
            "title": submission.title,
            "selftext": submission.selftext,
            "score": submission.score,
            "url": submission.url,
            "num_comments": submission.num_comments,
            "created": submission.created,
            "author": str(submission.author)
        })
    df = pd.DataFrame(posts)
    return df

# Funktion: Abrufen von Posts aus mehreren Subreddits
def get_posts_from_multiple_subreddits(subreddit_list, keyword, limit=1000):
    all_posts = []
    for subreddit_name in subreddit_list:
        print(f"Fetching posts from r/{subreddit_name}...")
        try:
            df = get_reddit_posts(subreddit_name, keyword, limit)
            all_posts.append(df)
        except Exception as e:
            print(f"Fehler beim Abrufen von r/{subreddit_name}: {e}")
    combined_df = pd.concat(all_posts, ignore_index=True)
    return combined_df

# Funktion: Textvorverarbeitung mit spaCy
def preprocess_text(text):
    if not text:
        return ""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Funktion: Erwähnungen zählen
def count_mentions(text, keywords):
    if not isinstance(text, str):
        return 0
    return sum(word in text.lower() for word in keywords)

# Funktion: Sentiment analysieren
def get_sentiment(text):
    if not isinstance(text, str):
        return 0
    return TextBlob(text).sentiment.polarity

# Subreddit-Liste und Einstellungen
subreddit_list = ["politics", "PoliticalDiscussion", "Conservative", "democrats", "republican"]
keyword = "election"
limit = 100

# Lade das spaCy-Modell
nlp = spacy.load("en_core_web_sm")

# Abrufen der Posts aus mehreren Subreddits
df_all_posts = get_posts_from_multiple_subreddits(subreddit_list, keyword, limit)

# Zeitstempel in ein lesbares Datum umwandeln
df_all_posts['created_date'] = pd.to_datetime(df_all_posts['created'], unit='s')

# Zeitraum: Kamalas Nominierung bis Wahltag (2024)
start_date = datetime(2024, 7, 31)  # Kamalas Nominierung
end_date = datetime(2024, 11, 5)    # Wahltag

# Filtere die Posts nach Zeitraum
filtered_posts = df_all_posts[(df_all_posts['created_date'] >= start_date) &
                               (df_all_posts['created_date'] <= end_date)]

# Textvorbereitung: Bereinigung und Tokenisierung
filtered_posts['clean_title'] = filtered_posts['title'].apply(preprocess_text)
filtered_posts['clean_selftext'] = filtered_posts['selftext'].apply(preprocess_text)

# Keywords für Erwähnungen
kamala_keywords = ["kamala", "harris"]
trump_keywords = ["trump", "donald"]

# Erwähnungen zählen
filtered_posts['kamala_mentions'] = filtered_posts['clean_title'].apply(lambda x: count_mentions(x, kamala_keywords)) + \
                                    filtered_posts['clean_selftext'].apply(lambda x: count_mentions(x, kamala_keywords))

filtered_posts['trump_mentions'] = filtered_posts['clean_title'].apply(lambda x: count_mentions(x, trump_keywords)) + \
                                   filtered_posts['clean_selftext'].apply(lambda x: count_mentions(x, trump_keywords))

# Sentiment für relevante Posts berechnen
filtered_posts['kamala_sentiment'] = filtered_posts['clean_title'].apply(get_sentiment) + \
                                     filtered_posts['clean_selftext'].apply(get_sentiment)

filtered_posts['trump_sentiment'] = filtered_posts['clean_title'].apply(get_sentiment) + \
                                    filtered_posts['clean_selftext'].apply(get_sentiment)

# Speichere die gefilterten Daten
filtered_posts.to_csv("filtered_reddit_posts_2024.csv", index=False)

# Visualisierung: Erwähnungen über die Zeit
mentions_over_time = filtered_posts.groupby('created_date').agg({
    'kamala_mentions': 'sum',
    'trump_mentions': 'sum'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(mentions_over_time['created_date'], mentions_over_time['kamala_mentions'], label='Kamala Harris')
plt.plot(mentions_over_time['created_date'], mentions_over_time['trump_mentions'], label='Donald Trump')
plt.title("Erwähnungen von Kamala Harris und Donald Trump über die Zeit (2024)")
plt.xlabel("Datum")
plt.ylabel("Anzahl der Erwähnungen")
plt.legend()
plt.show()

# Visualisierung: Sentiment-Verteilung
plt.figure(figsize=(10, 6))
plt.hist(filtered_posts[filtered_posts['kamala_mentions'] > 0]['kamala_sentiment'], bins=20, alpha=0.5, label='Kamala Harris')
plt.hist(filtered_posts[filtered_posts['trump_mentions'] > 0]['trump_sentiment'], bins=20, alpha=0.5, label='Donald Trump')
plt.title("Sentiment-Verteilung für Kamala Harris und Donald Trump (2024)")
plt.xlabel("Sentiment-Wert")
plt.ylabel("Anzahl der Posts")
plt.legend()
plt.show()

# Ergebnisse ausgeben
kamala_total_mentions = filtered_posts['kamala_mentions'].sum()
trump_total_mentions = filtered_posts['trump_mentions'].sum()
kamala_avg_sentiment = filtered_posts[filtered_posts['kamala_mentions'] > 0]['kamala_sentiment'].mean()
trump_avg_sentiment = filtered_posts[filtered_posts['trump_mentions'] > 0]['trump_sentiment'].mean()

print(f"Erwähnungen von Kamala Harris: {kamala_total_mentions}")
print(f"Erwähnungen von Donald Trump: {trump_total_mentions}")
print(f"Durchschnittliches Sentiment für Kamala Harris: {kamala_avg_sentiment}")
print(f"Durchschnittliches Sentiment für Donald Trump: {trump_avg_sentiment}")
