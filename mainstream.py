#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install streamlit


# In[7]:


#pip! install streamlit-option-menu


# In[8]:



import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import time

import pickle
import re
import time


# In[ ]:


with st.sidebar:
    selected = option_menu ('Menu',
    ['Raw Data',
    'Preprocessing Data',
    'Support Vector Machine Model',
    'Naive Bayes Model',
    'Support Vector Machine Predict',
    'Naive Bayes Predict'],
     default_index=0)


# In[ ]:


Data = pd.read_csv('D:\SEMESTER 7\SKRIPSI\program\clean_teks_sastrawi.csv')
X = Data.clean_teks.values
y = Data.Sentimen.values

# Membuat empty List
processed_tweets = []

for tweet in range(0, len(X)):
    # Hapus semua special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))

    # Hapus semua single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # Hapus single characters dari awal
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

    # Substitusi multiple spaces dengan single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # Hapus prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    # Ubah menjadi Lowercase
    processed_tweet = processed_tweet.lower()

    # Masukkan ke list kosong yang telah dibuat sebelumnya
    processed_tweets.append(processed_tweet)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('indonesian'),ngram_range=(1,3))
X1 = tfidfconverter.fit_transform(processed_tweets).toarray()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.1, random_state=0) #data test 5%, data train =95%


# In[ ]:


from sklearn.svm import SVC
text_classifier_svm = SVC(kernel='rbf', C=0.4, gamma = 1)
t0_svm = time.time()
text_classifier_svm.fit(X_train, y_train)
t1_svm = time.time()

predictions_svm = text_classifier_svm.predict(X_test)
t2_svm = time.time()
time_linear_train_svm = t1_svm-t0_svm
time_linear_predict_svm = t2_svm-t1_svm




from sklearn.naive_bayes import GaussianNB
text_classifier_nb = GaussianNB()
t0_nb = time.time()
text_classifier_nb.fit(X_train, y_train)
t1_nb = time.time()

predictions_nb = text_classifier_nb.predict(X_test)
t2_nb = time.time()
time_linear_train_nb = t1_nb-t0_nb
time_linear_predict_nb = t2_nb-t1_nb




# In[ ]:





# In[ ]:


if (selected == 'Raw Data'):
    st.title('Raw Data Tren Bisnis Dari Komentar Pengguna Twitter')
    Raw = pd.read_csv('D:\SEMESTER 7\SKRIPSI\program/clean_teks_sastrawi.csv')
    st.write(Raw.head(15))
    
if (selected == 'Preprocessing Data'):
    st.title('Preprocessing Data Tren Bisnis Dari Komentar Pengguna Twitter')
    st.write(Data.head(15))
    
if (selected == 'Support Vector Machine Model'):
    st.title('Hasil Evaluasi Model Klasifikasi Support Vector Machine')
    st.write('Accuracy  = ', round(accuracy_score(y_test, predictions_svm)*100,2),'%')
    st.write("")
    st.success(f"SVM Training time: %fs" % (time_linear_train_svm))
    st.success(f"SVM Predict time: %fs" % (time_linear_predict_svm))
    st.write("")
    st.write("")
    st.write("Support Vector Machine")
    st.write(confusion_matrix(y_test,predictions_svm))
    st.write("")
    st.write("")

    
if (selected == 'Naive Bayes Model'):
    st.title('Hasil Evaluasi Model Klasifikasi Naive Bayes')
    st.write("")
    st.write("Naive Bayes")
    st.write('Accuracy  = ', round(accuracy_score(y_test, predictions_nb)*100,2),'%')
    st.write("")
    st.success(f"NB Training time: %fs" % (time_linear_train_nb))
    st.success(f"Prediction time: %fs" % (time_linear_predict_nb))
    st.write("")
    st.write("Naive Bayes")
    st.write(confusion_matrix(y_test,predictions_nb))
    st.write("")
    st.write("")
    
    
if (selected == 'Support Vector Machine Predict'):
    st.title('Prediksi Klasifikasi Support Vector Machine')
    review = st.text_input("Inputkan teks")

    review_vector = tfidfconverter.transform([review]).toarray() # vectorizing
    pred_text = text_classifier_svm.predict(review_vector)
    prediksi = st.button ("Klasifikasi")
    
    if prediksi :
        st.success (f'Hasil Klasifikasi{pred_text}')

    
if (selected == 'Naive Bayes Predict'):
    st.title('Prediksi Klasifikasi Naive Bayes')
    review = st.text_input("Inputkan teks")

    review_vector = tfidfconverter.transform([review]).toarray() # vectorizing
    pred_text = text_classifier_nb.predict(review_vector)

    prediksi = st.button ("Klasifikasi")
    
    if prediksi :
        st.success (f'Hasil Klasifikasi{pred_text}')

