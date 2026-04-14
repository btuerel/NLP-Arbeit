# Artikelzuweisung von Englischen 'Loanwords' (Lehnwörtern) im deutschen Sprachgebrauch

In dieser Arbeit habe ich recherchiert, wie Personen mit verschiednen Muttersprachen Lehnwörter aus dem Englischen im Deutschen nutzen un welche Artikel sie den Englischen Wörtern zuweisen.

## Forschungsfrage 

Unterscheidet sich die Nutzung der Artikel für englische Lehnwörter im deutschen Sprachgebrauch je nach Muttersprache?

## Daten

Das Datenset basiert auf einer schriftlichen Umfrage, welche mit 28 Personen durchgeführt wurde.
Die Umfrage enthielt: 

- Artikelzuweisungs Aufgabe, die Zuweisung von deutschen Artikeln für englische Wörter in einem deutschen Kontext 
- Linguistische Hintergrundinformation der Person 
- Ein Cloze-Test

Ich habe alle Daten manuell in die Excell Datei eingetragen und sie in Python verarbeitet.

## Dateien

Enthalten sind die Dateien:
- 'analysis.py' das analyse Script 
- 'analysis_dataset.csv' ein sauberes Datenset für die Analyse 
- 'survey_results.xlsx' das original strukturierte Datenset
- 'Figure_1.png' Artikelverteilung bei englischen Wörtern 
- 'Figure_2.png' Artikelwahl nach Sprachfamilie 
- 'ConfusionMatrix1.png' die Confusion Matrix des NLP Models 

## Methode 

Die Analyse besteht aus zwei Teilen:

1. Beschreibung 
 - allgemeine Artikelverteilung 
 - vergleich zwischen Sprachfamilien 

2. NLP-Experiment 
 - character n-gram Funktion 
 - das logistische Regressionsmodel 
 - Trainings- und Testdaten (80/20)
