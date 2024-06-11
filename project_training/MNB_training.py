#training the model

import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

df_train = pd.read_csv("project_data/cleaned_train_data_2.csv")

X_train, X_test, y_train, y_test = train_test_split(df_train['tweet'], df_train['label'], test_size=0.1,random_state=69)

vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

model_filename = 'project_models/model_MNB_1.joblib'
dump(clf, model_filename)

vectorizer_filename = 'project_models/tfidf_vectorizer_1.joblib'
dump(vectorizer, vectorizer_filename)

y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="spring", xticklabels=['False', 'True'], yticklabels=['False', 'True'])
# plt.show()



