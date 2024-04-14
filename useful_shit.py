#puzzles to be used 

# merging and correcting lists generated by chat 

# grand_list = []
# for i in range(3):  
#     true_tweets = eval(f"true_tweets_additional_{i}")
#     fake_tweets = eval(f"fake_tweets_additional_{i}")
#     grand_list.extend([(tweet, 1) for tweet in true_tweets])
#     grand_list.extend([(tweet, 0) for tweet in fake_tweets])
# print(len(grand_list))
# import re  
# def remove_hashtags(tweet):
#     return re.sub(r"\s*#\w+", "", tweet)
# grand_list = [(remove_hashtags(tweet), label) for tweet, label in grand_list]
# import csv
# csv_filename = 'NO2_fabricated_tweets_dataset.csv'
# with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['tweet', 'label'])
#     writer.writerows(grand_list)
# print(f'CSV file has been created successfully: {csv_filename}')


# # IMPORTANT!!!! use it to clean before placing data to train/test model

# import nltk
# import re
# from bs4 import BeautifulSoup

# nltk.download("stopwords")  
# nltk.download('punkt')      
# nltk.download('wordnet')    

# from nltk.corpus import stopwords

# def remove_html(text):
#     soup = BeautifulSoup(text, "html.parser")
#     return soup.get_text()

# def remove_punctuations(text):
#     return re.sub(r'\[[^]]*\]', '', text)

# def remove_characters(text):
#     return re.sub(r"[^a-zA-Z]", " ", text)

# def remove_stopwords_and_lemmatization(text):
#     final_text = []
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     lemma = nltk.WordNetLemmatizer()  
#     stop_words = set(stopwords.words('english'))  
#     for word in text:
#         if word not in stop_words:
#             word = lemma.lemmatize(word)
#             final_text.append(word)
#     return " ".join(final_text)

# def cleaning(text):
#     text = remove_html(text)
#     text = remove_punctuations(text)
#     text = remove_characters(text)
#     text = remove_stopwords_and_lemmatization(text)
#     return text


#bylo wiecej ale bardziej research i testowanie a te sa takie co sie przydza sie 