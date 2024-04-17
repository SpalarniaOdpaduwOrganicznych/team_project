from joblib import load
from useful_tools import cleaning

clf_loaded = load("model_MNB.joblib")
vectorizer_loaded = load("tfidf_vectorizer_1.joblib")

print("paste the concerning tweet here: ")
usr_input = input()
tweet = cleaning(usr_input)

X_test_transformed = vectorizer_loaded.transform([tweet])
predictions = clf_loaded.predict(X_test_transformed)

print("0 to fake a 1 true jak cos ")
print(predictions)
#a to to jest zeby se sprawdzic jak sie cleaning zrobil i czy funkcje clean nie poprawic jakos (juz widze ze trzeba bedzie)
print(tweet)



