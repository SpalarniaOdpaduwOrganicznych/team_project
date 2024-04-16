''' 
test na 180 tweetach wygenerowanych przez chat, 
tweety olabelowane (wygenerowane z labelem), 
w pliku znajduja sie juz oczyszczone dane tekstowe skryptem, 
ktory znajdziecie w useful_shit.py


accuracy: 0.7777777777777778
aconfusion mtrix :
 [[64 26]
 [14 76]]
C. Report :
               precision    recall  f1-score   support

           0       0.82      0.71      0.76        90
           1       0.75      0.84      0.79        90

    accuracy                           0.78       180
   macro avg       0.78      0.78      0.78       180
weighted avg       0.78      0.78      0.78       180

wiadomo wynik do poprawy ale przynajmej 
jak sie pani zapyta czy cos mamy to mozemy powiedziec ze 'prawie'

wgl moglem dac to w readme normalnie w repo a nie taki cyrk w tych komentarzach tak teraz mysle///

'''
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

clf_loaded = load("model_MNB.joblib")
vectorizer_loaded = load("tfidf_vectorizer_1.joblib")

testing_df = pd.read_csv("df_5.csv")

if 'tweet' in testing_df.columns and 'label' in testing_df.columns:
    X_test_transformed = vectorizer_loaded.transform(testing_df['tweet'])
    predictions = clf_loaded.predict(X_test_transformed)

    accuracy = accuracy_score(testing_df['label'], predictions)
    conf_matrix = confusion_matrix(testing_df['label'], predictions)
    class_report = classification_report(testing_df['label'], predictions)

    print("accuracy:", accuracy)
    print("aconfusion mtrix :\n", conf_matrix)
    print("C. Report :\n", class_report)
else:
    print("Error: Dataframe does not exist")

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='jet', xticklabels=clf_loaded.classes_, yticklabels=clf_loaded.classes_)
plt.show()
