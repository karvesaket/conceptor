'''Trains an LSTM model on the IMDB sentiment classification task.
Adapted from 
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from collections import defaultdict
import keras
import numpy as np
import os
import gensim
from gensim.models.keyedvectors import KeyedVectors

efolder = '/content/'
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

embeddings_index = {}
embd_type = input("Enter embedding type")
conceptor_flag = input("Conceptor?")
if embd_type is "glove":
  resourceFile = '/content/'
  currembd = KeyedVectors.load_word2vec_format(resourceFile + 'gensim_glove.840B.300d.txt.bin', binary=True)
elif embd_type is "word2vec":
  resourceFile = '/content/'
  currembd = KeyedVectors.load_word2vec_format(resourceFile + 'GoogleNews-vectors-negative300.bin', binary=True)                       
print('The embedding has been loaded from gensim!')
#epath = efolder + efile

#with open(epath) as f:
#  for line in f:
#    values = line.split(' ')
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#embedding_dim = coefs.shape[0]
#print(embedding_dim)
embedding_dim = currembd.vectors.shape[1]
print(embedding_dim)
index_dict = keras.datasets.imdb.get_word_index()
n_vocab = len(index_dict) + 2
print("n_vocab", n_vocab)
oov_count = 0
embedding_weights = np.zeros((n_vocab, embedding_dim))
for word, index in index_dict.items():
    word = word.lower()
    if word in currembd:
        embedding_weights[index,:] = currembd[word]
    else:
        oov_count += 1
        embedding_weights[index,:] = currembd['unk']

if conceptor_flag is "y":
  print("conceptoring")
  embedding_weights = post_process_cn_matrix(embedding_weights) 
  print("conceptored!!")
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(n_vocab, embedding_dim, weights=[embedding_weights],trainable=False))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
