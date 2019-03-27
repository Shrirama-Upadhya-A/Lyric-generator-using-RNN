# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:28:29 2018

@author: tanma
"""

# Importing Modules
import numpy as np
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Bidirectional
from keras.layers import CuDNNLSTM,GlobalMaxPool1D

# Importing Data
data = pd.read_csv('drake-songs.csv')

text = ''

# Cleaning Data
for index, row in data['lyrics'].iteritems():
    cleaned = str(row).lower().replace(' ', '\n').replace('|-|','\n')
    text = text + " ".join(re.findall(r"[a-z']+", cleaned))
    
tokens = re.findall(r"[a-z'\s]", text)

chars = sorted(list(set(tokens)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

vocab_size = len(chars)

# Vectorizing
maxlen = 40
step = 1
sentences = []
next_char = []
for i in range(0, len(text)-maxlen, step):
  sentences.append(text[i:i+maxlen])
  next_char.append(text[i+maxlen])

# One Hot Encoding  
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)))
  
for i, sentence in enumerate(sentences):
  for j, char in enumerate(sentence):
    x[i, j, char_indices[char]] = 1
  y[i, char_indices[next_char[i]]] = 1


# Model Construction
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(256,input_shape = (maxlen,vocab_size),return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences = True)))
model.add(Dropout(0.2))
model.add(GlobalMaxPool1D())
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = "nadam",metrics = ['accuracy'])

# Model Fitting
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor = "loss", verbose = 1,save_best_only = True,mode = "min")
callbacks_list = [checkpoint]

model.fit(x, y, batch_size = 128, epochs = 10,callbacks = callbacks_list)  

# Model Performance
model.summary()

# Saving the Model
model.save('model.h5')


   
		    