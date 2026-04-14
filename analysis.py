import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Daten laden
participants = pd.read_excel("survey_results.xlsx", sheet_name="participants")
responses = pd.read_excel("survey_results.xlsx", sheet_name="word_article_responses")

# Spaltennamen bereinigen
participants.columns = participants.columns.str.strip()
responses.columns = responses.columns.str.strip()

# Tabellen zusammenführen
df = responses.merge(participants, on="participant_id")

# Sprachfamilien definieren
language_map = {
    "German": "Germanic",
    "English": "Germanic",
    "Dutch": "Germanic",

    "Greek": "Indo-European",
    "Russian": "Indo-European",
    "Persian": "Indo-European",
    "Kurdish": "Indo-European",
    "Pashto": "Indo-European",
    "Singhalese": "Indo-European",

    "Turkish": "Turkic",

    "Arabic": "Afro-Asiatic",
    "Amharic": "Afro-Asiatic",

    "Lingala": "Niger-Congo"
}

# neue Spalte erstellen
df["language_family"] = df["language_group"].map(language_map)
df["language_family"] = df["language_family"].fillna("Other")

# Artikel bereinigen
df["word_article"] = df["word_article"].astype(str).str.strip().str.lower()

# Wörter bereinigen
df["word"] = df["word"].astype(str).str.strip().str.lower()

# nur gültige Artikel behalten
df = df[df["word_article"].isin(["der", "die", "das"])]

# Datensatz speichern
df.to_csv("analysis_dataset.csv", index=False)

# Ausgabe
print("Datensatzgröße:")
print(df.shape)

print("\nArtikelverteilung:")
print(df["word_article"].value_counts())

print("\nVerwendete Artikelwerte:")
print(sorted(df["word_article"].unique()))

print("\nSprachgruppen und Sprachfamilien:")
print(df[["language_group", "language_family"]].drop_duplicates().sort_values("language_group"))

# Reihenfolge festlegen
article_order = ["der", "die", "das"]
family_order = ["Germanic", "Indo-European", "Turkic", "Afro-Asiatic", "Niger-Congo", "Other"]

# Grafik 1: Gesamtverteilung
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="word_article", order=article_order)

plt.title("Artikelverteilung bei englischen Wörtern")
plt.xlabel("Artikel")
plt.ylabel("Häufigkeit")
plt.tight_layout()
plt.savefig("article_distribution.png", dpi=300)
plt.show()

# Grafik 2: Sprachfamilien
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="language_family",
    order=family_order,
    hue="word_article",
    hue_order=article_order
)

print("\nArtikel nach Sprachfamilie (Prozent):")
table_percent = pd.crosstab(
    df["language_family"],
    df["word_article"],
    normalize="index"
) * 100

print(table_percent.round(1))

plt.title("Artikelwahl nach Sprachfamilie")
plt.xlabel("Sprachfamilie")
plt.ylabel("Häufigkeit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("article_by_language_family.png", dpi=300)
plt.show()


# NLP-Teil

print("\n--- NLP Experiment ---")

X_words = df["word"]
y = df["word_article"]

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_words, y, test_size=0.2, random_state=42, stratify=y
)

# Vektorisierung
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2,4))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modell trainieren
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Vorhersagen auf TESTDATEN
y_pred = model.predict(X_test_vec)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix berechnen
cm = confusion_matrix(y_test, y_pred, labels=["der", "die", "das"])

# Visualisierung
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["der", "die", "das"])
disp.plot(cmap="Blues")

plt.title("Confusion Matrix: Artikelvorhersage")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()