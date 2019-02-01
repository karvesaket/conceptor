"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)
"""
from __future__ import print_function
import numpy as np
import gzip
import os
from gensim.models.keyedvectors import KeyedVectors

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

#path where the pretrained embedding file is stored
efolder = '/content/'
efile = input("Enter embeddings file name: ")
epath = efolder + efile

#Train, Dev, and Test files
folder = 'data/'
files = [folder+'train.txt',  folder+'dev.txt', folder+'test.txt']

def post_process_cn_matrix(x, alpha = 2):
    print("starting...")
    #x = orig_embd.vectors
    print(x.shape)

    #Calculate the correlation matrix
    R = x.dot(x.T)/(x.shape[1])
    print("R calculated")
    #Calculate the conceptor matrix
    C = R @ (np.linalg.inv(R + alpha ** (-2) * np.eye(x.shape[0])))
    print("C calculated")
    
    #Calculate the negation of the conceptor matrix
    negC = np.eye(x.shape[0]) - C
    print("negC calculated")
    
    #Post-process the vocab matrix
    newX = (negC @ x).T
    print(newX.shape)
    return newX




def createMatrices(sentences, word2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']


    xMatrix = []
    unknownWordCount = 0
    wordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        sentenceWordIdx = []

        for word in sentence:
            wordCount += 1

            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            sentenceWordIdx.append(wordIdx)

        xMatrix.append(sentenceWordIdx)


    print("Unknown tokens: %.2f%%" % (unknownWordCount/(float(wordCount))*100))
    return xMatrix

def readFile(filepath):
    sentences = []
    labels = []

    for line in open(filepath):
        splits = line.split()
        label = int(splits[0])
        words = splits[1:]

        labels.append(label)
        sentences.append(words)

    print(filepath, len(sentences), "sentences")

    return sentences, labels






# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl/data.pkl.gz'


trainDataset = readFile(files[0])
devDataset = readFile(files[1])
testDataset = readFile(files[2])


# :: Compute which words are needed for the train/dev/test set ::
words = {}
for sentences, labels in [trainDataset, devDataset, testDataset]:
    for sentence in sentences:
        for token in sentence:
            words[token.lower()] = True


# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []


# :: Load the pre-trained embeddings file ::
#fEmbeddings = open(epath)
embd_type = input("Enter embedding type")
if embd_type == "glove":
    resourceFile = '/content/'
    curr_embd = KeyedVectors.load_word2vec_format(resourceFile + 'gensim_glove.840B.300d.txt.bin', binary=True)
    print(curr_embd.vectors.shape[1])
elif embd_type == "word2vec":
    resourceFile = '/content/'
    curr_embd = KeyedVectors.load_word2vec_format(resourceFile + 'GoogleNews-vectors-negative300.bin', binary=True)                       
print('The embedding has been loaded from gensim!')

print("Load pre-trained embeddings file")
embd_dim = curr_embd.vectors.shape[1]
for word in words:
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(embd_dim) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = curr_embd['unk']
        wordEmbeddings.append(vector)
    if word in curr_embd:
        vector = curr_embd[word]
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))
con = input("Conceptor? ")
if con == "y":
    wordEmbeddings_new = post_process_cn_matrix(wordEmbeddings.T)
else:
    wordEmbeddings_new = wordEmbeddings


# :: Create matrices ::
train_matrix = createMatrices(trainDataset[0], word2Idx)
dev_matrix = createMatrices(devDataset[0], word2Idx)
test_matrix = createMatrices(testDataset[0], word2Idx)


data = {
    'wordEmbeddings': wordEmbeddings_new, 'word2Idx': word2Idx,
    'train': {'sentences': train_matrix, 'labels': trainDataset[1]},
    'dev':   {'sentences': dev_matrix, 'labels': devDataset[1]},
    'test':  {'sentences': test_matrix, 'labels': testDataset[1]}
    }


f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")
