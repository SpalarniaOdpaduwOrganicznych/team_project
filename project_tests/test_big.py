from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

clf_loaded = load("project_models/model_MNB_1.joblib")
vectorizer_loaded = load("project_models/tfidf_vectorizer_1.joblib")

testing_df = pd.read_csv("project_data/test_data_1.csv")

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
