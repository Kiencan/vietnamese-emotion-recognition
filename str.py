
import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_df = pd.read_excel('./data/train_nor_811.xlsx')
validation_df = pd.read_excel('./data/valid_nor_811.xlsx')
test_df = pd.read_excel('./data/test_nor_811.xlsx')

texts = pd.concat([train_df['Sentence'], validation_df['Sentence'], test_df['Sentence']])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

label = [ 'Anger', 'Disgust','Enjoyment','Fear', 'Other', 'Sadness' ,'Surprise']

max_sequence_length = 100
X = pad_sequences(sequences, maxlen=max_sequence_length)

st.title('Demo Classification by text')

input = st.text_input("Nhập một câu")

input = list(input)
if input != []:

    sequences_ = tokenizer.texts_to_sequences(input)
    X_ = pad_sequences(sequences_, maxlen=max_sequence_length)

    saved_model = load_model('best_model_cnn_w2v.keras')

    y = saved_model.predict(X_)
    y_pre = np.argmax(y, axis=1)

    st.write(label[y_pre[0]])


else: st.write(" ")





