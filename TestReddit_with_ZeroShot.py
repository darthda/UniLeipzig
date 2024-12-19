import praw
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import pipeline
import time

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
subreddit_list = ["politics", "PoliticalDiscussion", "Conservative", "democrats", "republican"]
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
