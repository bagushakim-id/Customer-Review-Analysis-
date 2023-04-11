import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import string 
import re
import pandas as pd
import numpy as np
import zipfile
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import TextVectorization
from keras.layers.preprocessing.text_vectorization import TextVectorization as KerasTextVectorization


@keras.utils.register_keras_serializable()
class CustomTextVectorization(TextVectorization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Dictionary of English Contractions
contractions_dict = { "ain't": "are not"}

# Load the saved model
zip_filename = 'customer_review_model.zip'

# Extract the zip file
with zipfile.ZipFile(zip_filename, 'r') as zip_obj:
    zip_obj.extractall()

with CustomObjectScope({'CustomTextVectorization': CustomTextVectorization, 'KerasTextVectorization': KerasTextVectorization}):
    model = load_model('customer_review_model')


def preprocess_review(review):
    review = ' '.join([contractions_dict.get(word, word) for word in review.split()])
    review = review.lower()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = word_tokenize(review)
    review = [word for word in review if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    review = pd.DataFrame([review])
    review.columns = ['stem_n_lemma']
    return review


def predict_sentiment(review):
    review = preprocess_review(review)
    y_pred = model.predict(review['stem_n_lemma'])
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    if y_pred[0] == 1:
        return "Positive"
    else:
        return "Negative"


def main():
    st.title("Customer Review Sentiment Analysis")
    review = st.text_input("Enter your review:")
    if review:
        processed_review = preprocess_review(review)
        st.write("Processed Review:", processed_review['stem_n_lemma'][0])
        result = predict_sentiment(review)
        st.write("The sentiment of the review is:", result)


if __name__ == "__main__":
    main()
