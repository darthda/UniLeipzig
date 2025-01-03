import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import json
import numpy as np
import glob

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# vis
import pyLDAvis
import pyLDAvis.gensim

# spacy
import spacy
from gensim.models import CoherenceModel
from nltk.corpus import stopwords


# Datei laden wenn nötig

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Funktion: Topic-Modelling und Visualisierung

# Bei den gefilterten posts soll nur der text der Posts verwendet werden!!
# das Argument names der topic Modelling Function setzt neben die Punkte im Graphen die einzelnen Wörter!!
# Die names müssen aus dem text rausgefiltert werden :)!!

cleaned_posts = filtered_posts['selftext']
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
topic_modelling_pipeline(cleaned_posts, example_names, true_k=3, max_features=50, plot_size=(8, 8))


#----------------
#LDA Topic Modeling

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

lemmatized_posts = lemmatization(posts)

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

vis

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
