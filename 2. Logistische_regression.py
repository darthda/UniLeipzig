import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

from Topic_modeling_series.td_tf_official import model

# Schritt 1: Vorverarbeitung der Daten

# Nur Posts mit den Klassifikationen "pro-Trump" oder "pro-Harris" berücksichtigen
filtered_posts = filtered_posts[filtered_posts['classification'].isin(['pro-Trump', 'pro-Harris'])]

# Erstellen des Textes (Titel und Text) und der Labels
X = filtered_posts['title'] + " " + filtered_posts['selftext']  # Text (Titel + Inhalt)
y = filtered_posts['classification'].apply(lambda x: 1 if x == 'pro-Trump' else 0)  # Labels: 1 = pro-Trump, 0 = pro-Harris

# Schritt 2: Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 3: Vektorisierung der Texte
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Maximal 1000 Features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# Model testen

# Funktion aus der Übung um Results der Logistischen Regression zu plotten
def test_model(y_true,y_pred,y_pred_proba):

  #lets use some metrics
  print("Accuracy: ", accuracy_score(y_test,y_pred))
  print("AUROC: ", roc_auc_score(y_test,y_pred_proba[:,1]))
  print("Confusion Matrix: \n", pd.DataFrame(confusion_matrix(y_test, y_pred)))

  # Calculate the false positive rate, true positive rate, and thresholds for ROC
  fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

  # Plot the ROC curve
  plt.plot(fpr, tpr, label="ROC Curve")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic (ROC) Curve")
  plt.legend(loc="best")
  plt.show()

# Schritt 4: Logistische Regression trainieren
m = LogisticRegression()
m.fit(X_train_vec, y_train)

# Schritt 5: Vorhersagen auf den Testdaten
y_pred = m.predict(X_test_vec)
y_pred_proba = m.predict_proba(X_test_vec)
test_model(y_test,y_pred,y_pred_proba)

# Schritt 6: Modellbewertung
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"]))

# Optional: Speichern des Modells und des Vektorisierers
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'count_vectorizer.pkl')

# Optional: Vorhersage für neue Texte
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "pro-Trump" if prediction == 1 else "pro-Harris"

# Beispiel für die Vorhersage eines neuen Posts
new_post = "Kamala Harris has been doing a great job!"
print(predict_sentiment(new_post))  # Ausgabe: pro-Harris

## Regulize the model with L1

# Schritt 1: Datenvorbereitung (bereits gegeben)
filtered_posts = filtered_posts[filtered_posts['classification'].isin(['pro-Trump', 'pro-Harris'])]
X = filtered_posts['title'] + " " + filtered_posts['selftext']
y = filtered_posts['classification'].apply(lambda x: 1 if x == 'pro-Trump' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 2: Vektorisierung der Texte
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Schritt 3: Logistische Regression mit L1-Regularisierung
m = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)  # C bestimmt die Regularisierungsstärke
m.fit(X_train_vec, y_train)

# Schritt 4: Vorhersagen
y_pred = m.predict(X_test_vec)
y_pred_proba = m.predict_proba(X_test_vec)

# Funktion zur Bewertung und Visualisierung
def test_model(y_true, y_pred, y_pred_proba):
    # Metriken
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("AUROC: ", roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred)))

    # ROC-Kurve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="best")
    plt.show()

test_model(y_test, y_pred, y_pred_proba)

# Schritt 5: Visualisierung der Koeffizienten
coefficients = m.coef_[0]
features = vectorizer.get_feature_names_out()

# Datenrahmen für die Visualisierung
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

# Positive und negative Koeffizienten anzeigen
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

# Schritt 6: Modellbewertung
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=["pro-Harris", "pro-Trump"]))

# Optional: Speichern des Modells und des Vektorisierers
import joblib
joblib.dump(m, 'logistic_regression_l1_model.pkl')
joblib.dump(vectorizer, 'count_vectorizer_l1.pkl')

# Beispielvorhersage für einen neuen Text
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = m.predict(text_vec)
    return "pro-Trump" if prediction == 1 else "pro-Harris"

# Beispiel für die Vorhersage
new_post = "Kamala Harris has been doing a great job!"
print(predict_sentiment(new_post))

