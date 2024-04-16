'''
model training stuff, 
tu jest po najnizszej linii oporu robione na MNB i TFV jako vectorizer, 
testowalem np SVM czy tez LSTMClassifier i nie dawalo to lepszych efektow,
do tego lstm uzywalem basic_english tokenizer - nienajlepszy jednym slowem,
ale imo dobor vectorizera do nn nie ma duzo sensu przy naszym rozmiarze danych (nie chcemy uzywac nn [chyba])

(mowielm ze zajebisty layout to classification reportdaje)

accuracy: 0.8961748633879781
confusion matrix:
 [[477  87]
 [ 27 507]]
               precision    recall  f1-score   support

           0       0.95      0.85      0.89       564
           1       0.85      0.95      0.90       534

    accuracy                           0.90      1098
   macro avg       0.90      0.90      0.90      1098
weighted avg       0.90      0.90      0.90      1098


wgl moglem dac readme normalnie w repo a nie taki cyrk w tych komentarzach tak teraz mysle///
'''

import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

df_train = pd.read_csv("df_4.csv")

X_train, X_test, y_train, y_test = train_test_split(df_train['tweet'], df_train['label'], test_size=0.1,random_state=69)

vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

model_filename = 'model_MNB.joblib'
dump(clf, model_filename)

vectorizer_filename = 'tfidf_vectorizer_1.joblib'
dump(vectorizer, vectorizer_filename)

y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
#awsome function i have discovered it recently and idk i was really amazed with it and i will use it everywhere everyday 
print("classification eeport:\n", classification_report(y_test, y_pred))


# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="spring", xticklabels=['False', 'True'], yticklabels=['False', 'True'])
# plt.show()



