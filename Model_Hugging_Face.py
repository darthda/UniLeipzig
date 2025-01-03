from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np

# 1. Modell und Tokenizer laden
MODEL_NAME = "bert-base-uncased"  # Ersetze dies durch dein eigenes Hugging Face Modell
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # 3 Labels: pro-Trump, pro-Harris, neutral

# 2. Deine gefilterten Posts vorbereiten
# Annahme: "filtered_posts" ist ein DataFrame mit einer Spalte "title" und Labeln "classification" für Training
def prepare_dataset(df, text_column, label_column):
    # Konvertiere DataFrame zu Hugging Face Dataset
    dataset = Dataset.from_pandas(df[[text_column, label_column]])
    dataset = dataset.rename_column(text_column, "text")
    dataset = dataset.rename_column(label_column, "label")
    return dataset

# Hier ein Beispiel: Verwende nur einen Teil der Daten für das Training
# Für deine Posts passe dies entsprechend an
filtered_posts["label"] = filtered_posts["classification"].map({
    "pro-Trump": 0,
    "pro-Harris": 1,
    "neutral": 2
})  # Mappe Labels auf Zahlen
dataset = prepare_dataset(filtered_posts, "title", "label")

# 3. Train-Test-Split
split_dataset = dataset.train_test_split(test_size=0.2)

# 4. Tokenisierung
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

# 5. Evaluation-Metriken laden
accuracy = load_metric("accuracy")
f1 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {**acc, **f1_score}

# 6. Trainings-Argumente
training_args = TrainingArguments(
    num_train_epochs=3,             # Anzahl der Epochen
    output_dir="./model_output",    # Verzeichnis für Ausgaben
    per_device_train_batch_size=16, # Batch-Größe für Training
    per_device_eval_batch_size=16,  # Batch-Größe für Evaluation
    evaluation_strategy="epoch",    # Evaluation nach jeder Epoche
    save_strategy="epoch",          # Speichern nach jeder Epoche
    logging_dir="./logs",           # Logging-Verzeichnis
    load_best_model_at_end=True,    # Bestes Modell am Ende laden
)

# 7. Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 8. Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9. Training
trainer.train()

# 10. Vorhersagen auf neuen Posts
def classify_posts(posts, model, tokenizer):
    inputs = tokenizer(posts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = np.argmax(logits.detach().numpy(), axis=-1)
    label_map = {0: "pro-Trump", 1: "pro-Harris", 2: "neutral"}
    return [label_map[p] for p in predictions]

# Vorhersagen für deine Posts
filtered_posts["predicted_classification"] = classify_posts(filtered_posts["title"].tolist(), model, tokenizer)

# 11. Speichern der Ergebnisse
filtered_posts.to_csv("classified_reddit_posts_huggingface.csv", index=False)

print("\nKlassifikation abgeschlossen. Ergebnisse gespeichert in 'classified_reddit_posts_huggingface.csv'.")