import praw
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pyLDAvis
from charset_normalizer import detect
from datasets import Dataset
import symspellpy
from gensim import corpora
from nltk.cluster import kmeans
from pyLDAvis import gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import CoherenceModel
from symspellpy import SymSpell, Verbosity
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import pipeline
import time
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt


# Benötigte libraries für Data Cleaning
import re
import string
from langdetect import detect
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud


nltk.download('punkt')
nltk.download('stopwords')
nltk.download("punkt:tab")



# Reddit-API einrichten
reddit = praw.Reddit(
    client_id="KQZHY2dGalJMF3zo20Ihkg",
    client_secret="y4u1OAupdHZhC2N-N4urFgVX2zcmCA",
    user_agent="US_Wahl_Analyse"
)

# GPU-basierte Pipeline initialisieren
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Funktion: Abrufen von Posts aus einem Subreddit
# Laden der Spacy-Pipeline
nlp = spacy.load("en_core_web_sm")


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

# Zero-Shot-Klassifikationsmodell

def classify_batch(batch):
    results = classifier(batch["title"], candidate_labels=["pro-Trump", "pro-Harris", "neutral"])
    batch["classification"] = [res["labels"][0] for res in results]
    batch["confidence"] = [res["scores"][0] for res in results]
    return batch

# Datenbereinigung: Pipeline-Funktionen

nlp = spacy.load("en_core_web_sm")


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


# Subreddits und Keyword festlegen
subreddit_list = ["politics", "PoliticalDiscussion", "democrats"]
keyword = "election"
max_posts = 5000

# Abrufen der Daten
all_posts = []
for subreddit in subreddit_list:
    print(f"Abrufen von Posts aus r/{subreddit}...")
    try:
        df = get_reddit_posts(subreddit, keyword, max_posts=max_posts)
        all_posts.append(df)
    except Exception as e:
        print(f"Fehler beim Abrufen von r/{subreddit}: {e}")

df_all_posts = pd.concat(all_posts, ignore_index=True)
df_all_posts['created_date'] = pd.to_datetime(df_all_posts['created'], unit='s')

start_date = datetime(2024, 7, 31)
end_date = datetime(2024, 11, 5)
filtered_posts = df_all_posts[(df_all_posts['created_date'] >= start_date) & (df_all_posts['created_date'] <= end_date)]

# Speichern der unverarbeiteten Texte zur Analyse
filtered_posts[['title', 'selftext']].to_csv("unprocessed_texts.csv", index=False)

# Bereinigung der Texte
# Ausgabe des gefilterten Volltextes
# filtered_posts['cleaned_text'] = filtered_posts['selftext'].dropna().apply(preprocess_text)
cleaned_posts = filtered_posts['cleaned_text'] = filtered_posts['selftext'].dropna().apply(lambda x: preprocess_text(x)['cleaned_text'])


# Klassifikation der Beiträge
filtered_posts = filtered_posts.dropna(subset=['cleaned_text'])
dataset = Dataset.from_pandas(filtered_posts)

batch_size = 16
classified_dataset = dataset.map(classify_batch, batched=True, batch_size=batch_size)
classified_posts = classified_dataset.to_pandas()

# Schritt: Logistische Regression implementieren
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import seaborn as sns
import joblib
from matplotlib import pyplot as plt

# Nur Posts mit den Klassifikationen "pro-Trump" oder "pro-Harris" berücksichtigen
filtered_posts = classified_posts[classified_posts['classification'].isin(['pro-Trump', 'pro-Harris'])]

# Erstellen des Textes (Titel und Text) und der Labels
X = filtered_posts['title'] + " " + filtered_posts['selftext']  # Text (Titel + Inhalt)
y = filtered_posts['classification'].apply(lambda x: 1 if x == 'pro-Trump' else 0)  # Labels: 1 = pro-Trump, 0 = pro-Harris

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vektorisierung der Texte
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistische Regression trainieren (ohne Regularisierung)
m = LogisticRegression()
m.fit(X_train_vec, y_train)

# Vorhersagen auf den Testdaten
y_pred = m.predict(X_test_vec)
y_pred_proba = m.predict_proba(X_test_vec)

# Bewertung
print(f"Accuracy (ohne Regularisierung): {accuracy_score(y_test, y_pred):.4f}")
print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"]))

# Modell und Vektorisierer speichern
joblib.dump(m, 'models/logistic_regression_model.pkl')
joblib.dump(vectorizer, 'models/count_vectorizer.pkl')

# Logistische Regression mit L1-Regularisierung
m_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
m_l1.fit(X_train_vec, y_train)

# Vorhersagen
y_pred_l1 = m_l1.predict(X_test_vec)
y_pred_proba_l1 = m_l1.predict_proba(X_test_vec)

# Bewertung
print(f"Accuracy (L1-Regularisierung): {accuracy_score(y_test, y_pred_l1):.4f}")
print("\nKlassifikationsbericht (L1):")
print(classification_report(y_test, y_pred_l1, target_names=["pro-Harris", "pro-Trump"]))

# Berechnung und Ausgabe der Testmetriken
def calculate_metrics(y_test, y_pred, y_pred_proba):
    precision = classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"], output_dict=True)["pro-Trump"]["precision"]
    recall = classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"], output_dict=True)["pro-Trump"]["recall"]
    f1 = classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"], output_dict=True)["pro-Trump"]["f1-score"]
    auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    auprc = auc(recall_values, precision_values)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # Plot AUROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    
    # Plot AUPRC
    plt.figure(figsize=(10, 6))
    plt.plot(recall_values, precision_values, label=f"AUPRC = {auprc:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

# Visualisierung der Koeffizienten
coefficients = m_l1.coef_[0]
features = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

# Positive und negative Koeffizienten
plt.figure(figsize=(10, 6))
sns.barplot(
    data=coef_df.head(10),
    y="Feature",
    x="Coefficient",
    palette="viridis"
)
plt.title("Top 10 Positive Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=coef_df.tail(10),
    y="Feature",
    x="Coefficient",
    palette="viridis"
)
plt.title("Top 10 Negative Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# Modell und Vektorisierer speichern (L1-Regularisierung)
joblib.dump(m_l1, 'models/logistic_regression_l1_model.pkl')
joblib.dump(vectorizer, 'models/count_vectorizer_l1.pkl')

# Vorhersagefunktion
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = m_l1.predict(text_vec)
    return "pro-Trump" if prediction == 1 else "pro-Harris"

# Beispiel für die Vorhersage
new_post = "Kamala Harris has been doing a great job!"
print(predict_sentiment(new_post))

# Visualisierung der Klassifikationsergebnisse
classification_counts = classified_posts['classification'].value_counts()
plt.figure(figsize=(10, 6))
classification_counts.plot(kind='bar')
plt.title("Verteilung der Klassifikationen (pro-Trump, pro-Harris, neutral)")
plt.xlabel("Kategorie")
plt.ylabel("Anzahl der Posts")
plt.xticks(rotation=0)
plt.show()

classified_posts.to_csv("classified_reddit_posts_2024_cleaned.csv", index=False)

# Topic Modeling mit KMeans und PCA

def topic_modelling_pipeline(cleaned_posts, names, true_k=5, max_features=100, plot_size=(50, 50)):
    # 1. Vektorisierung mit TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        max_df=0.8,
        min_df=5,
        ngram_range=(1, 3),
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(cleaned_posts)

    # 2. KMeans-Clustering
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=10, random_state=42)
    model.fit(vectors)
    kmean_indices = model.predict(vectors)


    # 3. PCA für Visualisierung
    pca = PCA(n_components=2, random_state=42)
    scatter_plot_points = pca.fit_transform(vectors.toarray())

    # Farben für die Cluster
    colors = ["r", "b", "c", "y", "m", "g", "orange", "pink", "purple", "brown"]
    cluster_colors = [colors[label] for label in kmean_indices]

    # 4. Visualisierung der Cluster
    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]

    fig, ax = plt.subplots(figsize=plot_size)
    ax.scatter(x_axis, y_axis, c=cluster_colors, s=50)

    # Annotieren der Punkte mit Namen
    for i, txt in enumerate(names):
        ax.annotate(txt[:10], (x_axis[i], y_axis[i]), fontsize=8)

    plt.title("Topic Clustering Visualization")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()

    # 5. Cluster-Wörter ausgeben
    feature_names = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    print("\nTop-Wörter für jedes Cluster:")
    for i in range(true_k):
        print(f"Cluster {i}:")
        for ind in order_centroids[i, :10]:  # Top 10 Wörter pro Cluster
            print(f" {feature_names[ind]}")
        print("\n")


# Beispielanwendung der Pipeline

# Beispiel-Daten
#example_posts = [
   # "I love machine learning and data science!",
   # "Deep learning is a subset of machine learning.",
    #"KMeans clustering is a great unsupervised learning technique.",
   # "Data visualization is key to understanding your data.",
    #"PCA is a powerful tool for dimensionality reduction."
#]

# Namen als Dummy-Daten
#example_names = [f"Post {i}" for i in range(len(example_posts))]

# Topic-Modelling ausführen
example_names = [f"Post {i}" for i in range(len(cleaned_posts))]
topic_modelling_pipeline(cleaned_posts, example_names, true_k=3, max_features=50, plot_size=(8, 8))


#LDA
def lemmatization(filtered_posts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for post in filtered_posts:
        doc = nlp(post)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

lemmatized_posts = lemmatization(cleaned_posts)

def gen_words(posts):
    final = []
    for post in posts:
        new = gensim.utils.simple_preprocess(post, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_posts)


id2word = corpora.Dictionary(data_words)

corpus = []
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

print(corpus[0][:20])

word = id2word[0][:1][0]
print(word)

lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=10,
                                   random_state=100,
                                   update_every=1,
                                   chunksize=100,
                                   passes=10,
                                   alpha='auto')

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model,
                              corpus,
                              id2word,
                              mds="mmds",        # Verwende t-SNE für die Dimensionreduktion odwer mmds
                              R=20,              # Zeige die 20 relevantesten Begriffe pro Thema
                              sort_topics=False, # Sortiere Themen nicht nach Häufigkeit
                              lambda_step=0.05,  # Schrittweite für die Relevanzsteuerung
                              plot_opts={"xlab": "X-Achse", "ylab": "Y-Achse"}) # Plot-Anpassungen



#BIGRAMS AND TRIGRAMS

bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100) # Initialiserung von Bigram und Trigram Modellen die häufige Wortkombinationen erkennen
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)   #Phraser erstellt effizientere Versionen der Modelle, die für die Transformation von Texten genutzt werden.
trigram = gensim.models.phrases.Phraser(trigram_phrases) #Funktionen make_bigrams und make_trigrams transformieren Texte in Bigram- oder Trigram-Form.

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

print(data_bigrams_trigrams[0][0:20])



#TF-IDF REMOVAL
from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams) # ordnet jedem Wort eindeutige ID zu

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts] #Liste von Bag-of-Words-Repräsentationen (BoW) jedes Dokuments.
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

# Filtering out common words

low_value = 0.03
words  = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = [] #reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow  # new bow with common words filtered out

#id2word = corpora.Dictionary(data_words)

#corpus = []
#for text in data_words:
    #new = id2word.doc2bow(text)
    #corpus.append(new)

#print (corpus[0][0:20])

#word = id2word[[0][:1][0]]
#print (word)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto")
test_doc = corpus[-1]

#create vector of topic frequencys

vector = lda_model[test_doc]
print (vector)

## Sort topics based on frequency

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    return (sub_li)
new_vector = Sort(vector)
print (new_vector)

lda_model.save("models/test_model.model")

new_model = gensim.models.ldamodel.LdaModel.load("models/test_model.model")



pyLDAvis.enable_notebook()
# vispyLDAvis.gensim.prepare(lda_model, corpus, id2word, mvis = ds="mmds", R=30)


# 1. Daten vorbereiten
# Angenommen, deine Daten enthalten die Features (vektorisiert) und Labels
# filtered_posts ist dein gefilterter Datensatz mit Text (vektorisiert) und Labels
X = filtered_posts['title'] + " " + filtered_posts['selftext']  # Text (Titel + Inhalt)
y = filtered_posts['classification'].apply(lambda x: 1 if x == 'pro-Trump' else 0)

# Text in numerische Features umwandeln
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Maximal 5000 häufigste Wörter
X = vectorizer.fit_transform(X).toarray()  # Features aus Titel + Inhalt

# Encode die Labels in numerische Werte
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. CatBoost-Modell initialisieren
catboost_model = CatBoostClassifier(
    iterations=500,            # Anzahl der Iterationen (Bäume)
    depth=6,                   # Tiefe der Bäume
    learning_rate=0.1,         # Lernrate
    loss_function='MultiClass',# Multiklass-Logloss für Klassifizierung
    eval_metric='Accuracy',    # Bewertungsmetrik
    random_seed=42,            # Zufällige Startwerte für Reproduzierbarkeit
    verbose=100                # Log-Ausgabe nach jeder 100 Iterationen
)

# 3. Cross-Validation durchführen
catboost_data = Pool(data=X_train, label=y_train)
cv_results = cv(
    pool=catboost_data,
    params=catboost_model.get_params(),
    fold_count=5,            # 5-fache Cross-Validation
    plot=True,               # Plot der Lernkurven
    verbose=False            # Kein detailliertes Logging
)

# 4. CatBoost-Modell trainieren
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

# 5. Vorhersagen treffen
y_pred = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)

# 6. Metriken berechnen
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# Precision-Recall-Werte berechnen
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])

# Durchschnittlicher Precision-Recall-Score (AUPRC) berechnen
average_precision = average_precision_score(y_test, y_pred_proba[:, 1])

# Ausgabe der Ergebnisse
print("=== CatBoost Modell Evaluierung ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {average_precision:.4f}")
print("\n=== Klassifizierungsbericht ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion-Matrix visualisieren
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Precision-Recall-Kurve plotten
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f"AUPRC = {average_precision:.4f}", color='b')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid()
plt.show()
