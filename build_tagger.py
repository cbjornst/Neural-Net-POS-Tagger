# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file
    data = open(train_file)
    data = data.read().splitlines()
    dictionary = {}
    chars = {'I': 0, 'n': 1, 'a': 2, 'O': 3, 'c': 4, 't': 5, '.': 6, '1': 7, '9': 8, 'r': 9, 'e': 10, 'v': 11, 'i': 12, 'w': 13, 'o': 14, 'f': 15, '`': 16, 'T': 17, 'h': 18, 'M': 19, 's': 20, 'p': 21, "'": 22, 'C': 23, 'g': 24, 'G': 25, 'd': 26, 'm': 27, '(': 28, 'R': 29, 'l': 30, 'z': 31, 'k': 32, 'S': 33, 'W': 34, 'y': 35, ',': 36, 'L': 37, 'u': 38, '&': 39, 'A': 40, ')': 41, 'b': 42, 'K': 43, 'H': 44, 'E': 45, '-': 46, 'x': 47, 'U': 48, '2': 49, '0': 50, '4': 51, 'B': 52, 'F': 53, 'N': 54, 'D': 55, 'q': 56, '5': 57, '8': 58, '7': 59, '%': 60, '{': 61, '}': 62, 'j': 63, '/': 64, '$': 65, '3': 66, 'Y': 67, ':': 68, 'P': 69, 'Q': 70, 'J': 71, 'V': 72, '?': 73, ';': 74, 'Z': 75, '6': 76, '#': 77, 'X': 78, '!': 79, '\\': 80, '*': 81, '=': 82, '@': 83}
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    V = 0
    for line in data:
        line = line.split(" ")
        for word in line:
            word1, word2 = word.rsplit("/", 1)
            word1 = str(word1)
            word2 = str(word2)
            if word1 not in dictionary:
                dictionary[word1] = V
                V += 1
    wordEmbeds = nn.Embedding(V, 15)
    charEmbeds = nn.Embedding(85, 15)
    conv = nn.Conv1d(1, 15, 45)
    lstm = nn.LSTM(30, 30, bidirectional=True)
    lstm.hidden = (torch.zeros(2, 1, 30), torch.zeros(2, 1, 30))
    for line in range(1):    
        line = data[0].split(" ")
        embeddings = None
        for word in line:
            word1, word2 = word.rsplit("/", 1)
            word1 = str(word1)
            w = wordEmbedding(dictionary, chars, word1, wordEmbeds, charEmbeds, conv)
            if embeddings is None:
                embeddings = torch.FloatTensor(w)
            else:
                embeddings = torch.cat((embeddings, w))
    probs, v = lstm(embeddings.view(len(line), 1, 30), lstm.hidden)
    lstm.hidden = v
    tags = nn.Linear(60, 45)
    result = tags(probs)
    probabilities = []
    for i in range(len(result)):
        probabilities += [[]]
        summation = 0
        for j in range(len(result[i][0])):
            summation += np.exp(result[i][0][j].item())
        for j in range(len(result[i][0])):
            probabilities[i] += [(np.exp(result[i][0][j].item())) / summation]
    print(probabilities)
    print('Finished...')
    
def wordEmbedding(dictionary, chars, word, wEmbeds, cEmbeds, conv):
    w1 = torch.LongTensor([dictionary[word]])
    e = wEmbeds(w1)
    ci = None
    for i in range(len(word)):
        cj = None
        for j in range(3):    
            if (i + j - 1) > 0 and (i + j - 1) < len(word):
                c = [chars[word[i + j - 1]]]
                c = torch.LongTensor(c)
            else:
                c = [84]
                c = torch.LongTensor(c)
            c = cEmbeds(c)
            if cj is not None:
                cj = torch.cat((cj, c), 1)
            else:
                cj = c
        cj = cj.unsqueeze(0)
        c = conv(cj)
        if ci is not None:
            ci = torch.cat((ci, c), 2)
        else:
            ci = c
    maximum = nn.MaxPool1d(len(word))
    w = maximum(ci)
    w = w.view([1, 15])
    w = torch.cat((e, w), 1)
    return w
   
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
