from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

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