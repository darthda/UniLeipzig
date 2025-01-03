import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# GloVe-Modell laden
print("Lade vortrainiertes Modell 'glove-twitter-200'...")
glove_model = api.load("glove-twitter-200")
print("Modell geladen.")

# 1. Cosine Similarity zwischen wichtigen Wörtern untersuchen
def cosine_similarity_words(word1, word2, model):
    try:
        vec1 = model[word1]
        vec2 = model[word2]
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return similarity
    except KeyError:
        return None

# Wichtige Wahlwörter
wahlwörter = ["Trump", "Harris", "election", "vote", "democracy", "freedom"]

print("\nCosine Similarity zwischen Wahlwörtern:")
for word1 in wahlwörter:
    for word2 in wahlwörter:
        if word1 != word2:
            sim = cosine_similarity_words(word1, word2, glove_model)
            if sim is not None:
                print(f"{word1} <-> {word2}: {sim:.4f}")

# 2. Wörter, die "Trump" und "Harris" am ähnlichsten sind
def most_similar_words(word, model, topn=10):
    try:
        return model.most_similar(word, topn=topn)
    except KeyError:
        return []

print("\nWörter ähnlich zu 'Trump':")
print(most_similar_words("Trump", glove_model))

print("\nWörter ähnlich zu 'Harris':")
print(most_similar_words("Harris", glove_model))

# 3. Posts vektorisieren
def get_post_vector(post, model):
    # Extrahiere Wörter aus dem Post, die im Modell vorhanden sind
    words = [word for word in post.split() if word in model]
    if words:
        # Mittelwert der Vektoren berechnen
        vectors = np.array([model[word] for word in words])
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Vektorisieren der gefilterten Posts
filtered_posts['vector'] = filtered_posts['title'].apply(lambda x: get_post_vector(x, glove_model))

# 4. Cosine Similarity zwischen Posts und einem Zielbegriff (z.B. "Trump" oder "Harris")
def similarity_to_target(vector, target_word, model):
    try:
        target_vector = model[target_word]
        return cosine_similarity([vector], [target_vector])[0][0]
    except KeyError:
        return None

# Ähnlichkeit zu "Trump" und "Harris" berechnen
filtered_posts['similarity_to_trump'] = filtered_posts['vector'].apply(lambda x: similarity_to_target(x, "Trump", glove_model))
filtered_posts['similarity_to_harris'] = filtered_posts['vector'].apply(lambda x: similarity_to_target(x, "Harris", glove_model))

# 5. Ergebnisse speichern
filtered_posts.to_csv("reddit_posts_with_similarities.csv", index=False)

print("\nÄhnlichkeitsberechnungen abgeschlossen. Ergebnisse gespeichert in 'reddit_posts_with_similarities.csv'.")