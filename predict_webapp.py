# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:16:30 2018

@author: tanma
"""
# Importing Modules
import numpy as np
import pandas as pd
import random
import re
import flask
import os.path
from flask_cors import CORS
from flask import request
from load import *
from keras.models import load_model
from flask import Response


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

# Predicting
global model,graph
model,graph = init()


app = flask.Flask(__name__)

CORS(app)

@app.route("/", methods=["GET","POST"])
def predict():
    generated = ''
    with graph.as_default():
    		start_index=random.randint(0,len(text)-maxlen-1)
    		sent=text[start_index:start_index+maxlen]
    		generated+=sent
    		for i in range(200):
    		    x_sample=generated[i:i+maxlen]
    		    x=np.zeros((1,maxlen,vocab_size))
    		    for j in range(maxlen):
    		        x[0,j,char_indices[x_sample[j]]]=1
    		    probs=model.predict(x)
    		    probs=np.reshape(probs,probs.shape[1])
    		    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    		    generated+=indices_char[ix]
    print('Returning prediction...')
    return '<!DOCTYPE html><html lang="en"><head><title>Drake Song Generator</title><link href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" rel="stylesheet"></head><body><div class="jumbotron"><h1 class="display-4">Drake Lyric Generator</h1><p class="lead">This is an Deep-Learning Model to generate Lyrics of the Famous Rapper Drake.</p><hr class="my-4"><p>It uses LSTM for time series prediction.</p><p>Tanmay Thakur</p><p>Rushi Varun</p></hr></div><div class="jumbotron"><h3 class="display-5">Start Text:</h3><div class="container"><p class="lead">'+sent+'</p></div><h3 class="display-5">Generated Text:</h3><div class="container"><p class="lead">'+generated+'</p></div></div></body></html>'

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	print('Starting app...')
	app.run(host='0.0.0.0', port=port)
