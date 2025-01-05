import praw
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import pipeline
from datasets import Dataset
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
import joblib
import torch

# Schritt 1: Reddit-API einrichten
reddit = praw.Reddit(
    client_id="KQZHY2dGalJMF3zo20Ihkg",
    client_secret="y4u1OAupdHZhC2N-N4urFgVX2zcmCA",
    user_agent="US_Wahl_Analyse"
)

# GPU-basierte Pipeline initialisieren
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Funktion: Abrufen von Posts aus einem Subreddit

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
            last_post = submission.id

            if len(posts) >= max_posts:
                break
        time.sleep(1)

    return pd.DataFrame(posts)

# Funktion: Batchweise Zero-Shot-Klassifikation anwenden
def classify_batch(batch):
    results = classifier(batch["title"], candidate_labels=["pro-Trump", "pro-Harris", "neutral"])
    batch["classification"] = [res["labels"][0] for res in results]
    batch["confidence"] = [res["scores"][0] for res in results]
    return batch

# Subreddits und Keyword festlegen
subreddit_list = ["politics", "PoliticalDiscussion", "Conservative", "democrats", "republican"]
keyword = "election"
max_posts = 5000

# Abrufen und Klassifikation der Daten
all_posts = []
for subreddit in subreddit_list:
    print(f"Abrufen von Posts aus r/{subreddit}...")
    try:
        df = get_reddit_posts(subreddit, keyword, max_posts=max_posts)
        all_posts.append(df)
    except Exception as e:
        print(f"Fehler beim Abrufen von r/{subreddit}: {e}")

# Alle Daten zusammenfassen
df_all_posts = pd.concat(all_posts, ignore_index=True)

# Zeitstempel in lesbares Datum umwandeln
df_all_posts['created_date'] = pd.to_datetime(df_all_posts['created'], unit='s')

# Zeitraum: 31. Juli 2024 bis 5. November 2024
start_date = datetime(2024, 7, 31)
end_date = datetime(2024, 11, 5)
filtered_posts = df_all_posts[(df_all_posts['created_date'] >= start_date) & (df_all_posts['created_date'] <= end_date)]

# DataFrame in Hugging Face Dataset umwandeln
dataset = Dataset.from_pandas(filtered_posts)

# Batchweise Klassifikation anwenden
batch_size = 16  # Passe die Batchgröße an den verfügbaren GPU-Speicher an
classified_dataset = dataset.map(classify_batch, batched=True, batch_size=batch_size)

# Konvertiere das Ergebnis zurück in einen DataFrame
classified_posts = classified_dataset.to_pandas()

# Visualisierung: Verteilung der Klassifikationen (gesamt)
classification_counts = classified_posts['classification'].value_counts()
plt.figure(figsize=(10, 6))
classification_counts.plot(kind='bar')
plt.title("Verteilung der Klassifikationen (pro-Trump, pro-Harris, neutral)")
plt.xlabel("Kategorie")
plt.ylabel("Anzahl der Posts")
plt.xticks(rotation=0)
plt.show()

# Visualisierung: Erwähnungen über die Zeit
classified_posts['pro_trump'] = classified_posts['classification'] == "pro-Trump"
classified_posts['pro_harris'] = classified_posts['classification'] == "pro-Harris"
mentions_over_time = classified_posts.groupby('created_date').agg({
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
classified_posts.to_csv("classified_reddit_posts_2024.csv", index=False)

# Beispielvorhersage für neuen Text
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "pro-Trump" if prediction == 1 else "pro-Harris"

# Beispiel für eine Vorhersage
new_post = "Kamala Harris has been doing a great job!"
print(predict_sentiment(new_post))
