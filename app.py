import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def transform_review(movie_review):
    review = re.sub('[^a-zA-Z]', ' ', movie_review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    return review    

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
model = pickle.load(open('sentiment_model.pkl','rb'))

st.title("Movie recommender by audiance review")

input_review = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_review = transform_review(input_review)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_review])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Recommended")
    else:
        st.header("Not Recommended")