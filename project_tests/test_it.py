from joblib import load
from useful_tools import cleaning

clf_loaded = load("project_models/model_MNB_1.joblib")
vectorizer_loaded = load("project_models/tfidf_vectorizer_1.joblib")

print("paste the concerning tweet here: ")
usr_input = input()
tweet = cleaning(usr_input)

X_test_transformed = vectorizer_loaded.transform([tweet])
predictions = clf_loaded.predict(X_test_transformed)

if predictions == 1:
        print(f"Given tweet:\n{tweet}\n with a high probability is True")
else: 
        print(f"Given tweet:\n{tweet}\n with a high probability is Fake")





