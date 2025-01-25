import praw
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import re
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# NLTK-Initialisierung
nltk.download('punkt')
nltk.download('stopwords')

# Reddit-API einrichten
reddit = praw.Reddit(
    client_id="KQZHY2dGalJMF3zo20Ihkg",
    client_secret="y4u1OAupdHZhC2N-N4urFgVX2zcmCA",
    user_agent="US_Wahl_Analyse"
)

# Zero-Shot-Klassifikationsmodelle
device = 0  # GPU verwenden, falls verfügbar
classifier1 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
classifier2 = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)


# Funktion: Abrufen von Reddit-Posts
def get_reddit_posts(subreddit_name, keyword, max_posts=20000):
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
        time.sleep(2)  # Delay, um die API nicht zu überlasten

    return pd.DataFrame(posts)


# Vorverarbeitungspipeline
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Entfernen von HTML-Tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Entfernen von Stopwörtern
    stops = set(stopwords.words("english"))
    words = word_tokenize(text)
    text = " ".join([word for word in words if word not in stops])
    # Entfernen von Sonderzeichen
    text = re.sub(r"[^a-zA-Z0-9.,!?\-\' ]", "", text)
    # Entfernen von nicht-ASCII-Zeichen
    text = text.encode("ascii", errors="ignore").decode()
    return text


# Klassifikation mit einem Zero-Shot-Modell
def classify_with_model(posts, classifier_model):
    posts = posts[posts["cleaned_text"].notnull() & (posts["cleaned_text"].str.strip() != "")]
    text_list = posts["cleaned_text"].tolist()

    results = classifier_model(text_list, candidate_labels=["pro-Trump", "pro-Harris", "neutral"])
    posts["classification"] = [res["labels"][0] for res in results]
    posts["confidence"] = [res["scores"][0] for res in results]
    return posts


# Zusätzliche Metriken und Visualisierung
def evaluate_models(df1, df2):
    labels1 = df1["classification"]
    labels2 = df2["classification"]

    accuracy = accuracy_score(labels1, labels2)
    kappa = cohen_kappa_score(labels1, labels2)
    report = classification_report(labels1, labels2, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nKlassifikationsbericht:")
    print(classification_report(labels1, labels2))

    conf_matrix = confusion_matrix(labels1, labels2, labels=["pro-Trump", "pro-Harris", "neutral"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["pro-Trump", "pro-Harris", "neutral"],
                yticklabels=["pro-Trump", "pro-Harris", "neutral"])
    plt.title("Confusion Matrix")
    plt.xlabel("Modell 2 (Predicted)")
    plt.ylabel("Modell 1 (True)")
    plt.show()

    plt.figure(figsize=(10, 6))
    df1["classification"].value_counts().plot(kind="bar", alpha=0.7, label="Model 1")
    df2["classification"].value_counts().plot(kind="bar", alpha=0.7, label="Model 2")
    plt.title("Klassifikationsverteilungen der beiden Modelle")
    plt.xlabel("Klassifikation")
    plt.ylabel("Anzahl der Posts")
    plt.legend()
    plt.show()


# Hauptfunktion
def main():
    subreddit_list = ["politics", "PoliticalDiscussion", "democrats"]
    keyword = "election"
    all_posts = []

    for subreddit in subreddit_list:
        print(f"Abrufen von Posts aus r/{subreddit}...")
        posts = get_reddit_posts(subreddit, keyword, max_posts=6666)  # Verteile 20.000 gleichmäßig auf drei Subreddits
        all_posts.append(posts)
        time.sleep(5)  # Zusätzlicher Delay zwischen Subreddits

    df_all_posts = pd.concat(all_posts, ignore_index=True)

    print("Vorverarbeitung der Texte...")
    df_all_posts["cleaned_text"] = df_all_posts["selftext"].dropna().apply(preprocess_text)
    df_all_posts = df_all_posts.dropna(subset=["cleaned_text"])

    if df_all_posts.empty:
        print("Keine gültigen Texte nach der Vorverarbeitung.")
        return

    print("Klassifikation mit Modell 1...")
    classified_posts_model1 = classify_with_model(df_all_posts.copy(), classifier1)

    print("Klassifikation mit Modell 2...")
    classified_posts_model2 = classify_with_model(df_all_posts.copy(), classifier2)

    print("Vergleich der Klassifikationen...")
    evaluate_models(classified_posts_model1, classified_posts_model2)

    classified_posts_model1.to_csv("classified_posts_model1.csv", index=False)
    classified_posts_model2.to_csv("classified_posts_model2.csv", index=False)


if __name__ == "__main__":
    main()
