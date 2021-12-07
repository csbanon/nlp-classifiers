"""
A Comprehensive Survey of Machine Learning Methods for Text Classification
Moazam Soomro, Carlos Santiago Bañón
CAP 6307, Fall '21

lstm_ssm.py
===========
Defines, trains, and evaluates the LSTM architecture with the SMS spam dataset.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, classification_report
from keras.models import Model, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pdb 


def create_model(vocab_len, max_seq_len):
    inputs = Input(name='inputs', shape=[max_seq_len])   # None, 150
    layer = Embedding(vocab_length + 1, 50, input_length=max_seq_len)(inputs) # None, 150, 50
    layer = LSTM(64)(layer)  # None, 64
    layer = Dense(256,name='FC1')(layer) # None, 256
    layer = Activation('relu')(layer) # None, 256
    layer = Dropout(0.5)(layer) # None, 256
    layer = Dense(1,name='out_layer')(layer) # None, 1
    layer = Activation('sigmoid')(layer) # None, 1
    model = Model(inputs=inputs,outputs=layer)
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(), metrics=['acc'])
    return model


df = pd.read_csv("./data/raw/spam.csv", encoding='latin-1')
df_train = pd.read_csv("./data/spam_train.csv", encoding='latin-1')
df_test = pd.read_csv("./data/spam_test.csv", encoding='latin-1')
df.head()

df.info()

fig, ax = plt.subplots()
sns.countplot(df.v1, ax=ax)
ax.set_xlabel('Label')
ax.set_title('Number of ham and spam messages')
plt.show()

X = df.loc[:, 'v2']
y = df.loc[:, 'v1']

X_train_data, X_test_data, y_train_labels, y_test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_data = df_train.text
X_test_data = df_test.text
print(X_train_data.shape)
print(X_test_data.shape)

sent_lens = []
for sent in X_train_data: 
    sent_lens.append(len(word_tokenize(sent)))
    
print(max(sent_lens))

sns.distplot(sent_lens, bins=10, kde=True)
plt.show()

max_sequence_length = 23

tok = Tokenizer()
tok.fit_on_texts(X_train_data.values)

vocab_length = len(tok.word_index)

print('No. of unique tokens(vocab_size): ', vocab_length)

X_train_sequences = tok.texts_to_sequences(X_train_data.values)
X_test_sequences = tok.texts_to_sequences(X_test_data.values)

print('No of sequences:', len(X_train_sequences))

print(X_train_sequences[:2])

X_train = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test = pad_sequences(X_test_sequences, maxlen=max_sequence_length)
X_train[:2]

import pdb
pdb.set_trace()
y_train_labels.values
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.fit_transform(y_test_labels)
print(y_train)

model = create_model(vocab_length, max_sequence_length)
model.summary()
filepath='model_with_best_weights.h5'
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),  #EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=5),
             ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1),
#              TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1)             
            ]


history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2, callbacks=callbacks)

dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
history_dict = history.history
print(history_dict.keys())



plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for accuracy.
plt.plot(history_dict['acc'])
plt.plot(history_dict['val_acc'])
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pdb.set_trace()