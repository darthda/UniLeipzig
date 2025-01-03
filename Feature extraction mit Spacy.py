import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# spaCy Modell laden
nlp = spacy.load("en_core_web_sm")

# Verarbeitung der Posts mit spaCy
# Schauen ob die Daten in richtigem Format vorliegen!!!!!!

all_posts_text = filtered_posts['title'] + " " + filtered_posts['selftext']  # Titel und Inhalte kombinieren
docs = list(nlp.pipe(all_posts_text, batch_size=100, disable=["parser", "tagger"]))  # Pipeline optimiert

# 1. Entitätserkennung (NER)
print("Entitäten in den Posts (Beispiele):")
for doc in docs[:5]:  # Zeige nur Beispiele für Ausgabezwecke
    print(f"\nPost: {doc.text[:200]}...")  # Truncate langer Texte
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

# 2. Entitäten sammeln
print("\nSammeln aller Entitäten aus allen Posts...")
all_entities = [ent.text for doc in docs for ent in doc.ents]
entity_counts = Counter(all_entities)

# Häufigste Entitäten visualisieren
print("Visualisiere die Top-Entitäten...")
entity_df = pd.DataFrame(entity_counts.items(), columns=["Entity", "Count"]).sort_values(by="Count", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(entity_df["Entity"].head(10), entity_df["Count"].head(10))
plt.title("Top 10 häufige Entitäten in den Posts")
plt.xlabel("Entität")
plt.ylabel("Häufigkeit")
plt.xticks(rotation=45)
plt.show()

# 3. Schlüsselwörter extrahieren
print("\nWichtige Schlüsselwörter in allen Posts sammeln...")
all_keywords = [token.text for doc in docs for token in doc if token.is_alpha and not token.is_stop]
keyword_counts = Counter(all_keywords)

# Visualisierung der häufigsten Keywords
print("Visualisiere die Top-Keywords...")
keyword_df = pd.DataFrame(keyword_counts.items(), columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(keyword_df["Keyword"].head(10), keyword_df["Count"].head(10))
plt.title("Top 10 Schlüsselwörter in den Posts")
plt.xlabel("Schlüsselwort")
plt.ylabel("Häufigkeit")
plt.xticks(rotation=45)
plt.show()

# 4. Themenmodellierung basierend auf den häufigsten Entitäten
topics = entity_df["Entity"].head(10).tolist()
print("\nPotentielle Themen basierend auf Entitäten:")
print(", ".join(topics))

# Optional: Entitäten-Visualisierung mit displacy (auf einer zufälligen Auswahl von Posts)
print("\nVisualisiere Entitäten mit displacy...")
for doc in docs[:5]:  # Zeige nur die ersten 5 Posts
    displacy.render(doc, style="ent", jupyter=True)