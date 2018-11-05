# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        txt = self.data[index][0]
        label = self.data[index][1]
        return txt, label

    def __len__(self):
        return len(self.data)

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file
    data = open(train_file)
    data = data.read().splitlines()
    dictionary = {}
    correctTags = []
    chars = {'I': 0, 'n': 1, 'a': 2, 'O': 3, 'c': 4, 't': 5, '.': 6, '1': 7, '9': 8, 'r': 9, 'e': 10, 'v': 11, 'i': 12, 'w': 13, 'o': 14, 'f': 15, '`': 16, 'T': 17, 'h': 18, 'M': 19, 's': 20, 'p': 21, "'": 22, 'C': 23, 'g': 24, 'G': 25, 'd': 26, 'm': 27, '(': 28, 'R': 29, 'l': 30, 'z': 31, 'k': 32, 'S': 33, 'W': 34, 'y': 35, ',': 36, 'L': 37, 'u': 38, '&': 39, 'A': 40, ')': 41, 'b': 42, 'K': 43, 'H': 44, 'E': 45, '-': 46, 'x': 47, 'U': 48, '2': 49, '0': 50, '4': 51, 'B': 52, 'F': 53, 'N': 54, 'D': 55, 'q': 56, '5': 57, '8': 58, '7': 59, '%': 60, '{': 61, '}': 62, 'j': 63, '/': 64, '$': 65, '3': 66, 'Y': 67, ':': 68, 'P': 69, 'Q': 70, 'J': 71, 'V': 72, '?': 73, ';': 74, 'Z': 75, '6': 76, '#': 77, 'X': 78, '!': 79, '\\': 80, '*': 81, '=': 82, '@': 83}
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    V = 0
    trainingData = []
    for line in data:
        line = line.split(" ")
        t1 = []
        t2 = []
        for word in line:
            word1, word2 = word.rsplit("/", 1)
            word1 = str(word1)
            t1 += [word1]
            t2 += [word2]
            word2 = str(word2)
            if word1 not in dictionary:
                dictionary[word1] = V
                V += 1
        trainingData += [(t1, t2)]
    lstm = nn.LSTM(30, 30, bidirectional=True)
    lstm.lstm = nn.LSTM(30, 30, bidirectional=True)
    lstm.batch_size = 64
    lstm.wordEmbeds = nn.Embedding(V, 15)
    lstm.charEmbeds = nn.Embedding(85, 15)
    lstm.conv = nn.Conv1d(1, 15, 45)
    lstm.linear = nn.Linear(60, 45)
    lstm.hidden = (torch.zeros(2, lstm.batch_size, 30), torch.zeros(2, lstm.batch_size, 30))
    lstm.forward = forward
    lstm.dictionary = dictionary
    lstm.chars = chars
    lossFunction = nn.BCELoss()
    optimizer = optim.SGD(lstm.parameters(), lr=0.1)
    start = time.clock()
    ln = 0
    trainingData = Dataset(trainingData)
    train_loader = DataLoader(trainingData, batch_size=lstm.batch_size)
    for iter, trainData in enumerate(train_loader):
        sentence, trainTags = trainData
        lstm.batch_size = len(sentence)
        correctTags = []
        for i in range(len(trainTags)):
            t = []
            for j in range(len(trainTags[i])):
                t += [tags.index(trainTags[i][j])]
            correctTags += [[t]]
        s = nn.Sigmoid()
        correctTags = s(torch.FloatTensor(np.asarray(correctTags)))
        ln += 1
        if ln % 1000 == 0:
            print(time.clock() - start)
        lstm.zero_grad()
        lstm.hidden = (torch.zeros(2, lstm.batch_size, 30), torch.zeros(2, lstm.batch_size, 30))
        modelTags = lstm.forward(lstm, sentence)
        print(modelTags)
        modelTags = torch.FloatTensor(np.asarray(modelTags))
        correctTags = torch.FloatTensor(np.asarray(trainTags))
        loss = lossFunction(modelTags, correctTags)
        loss = Variable(loss, requires_grad = True)
        loss.backward()
        optimizer.step()
        torch.save(lstm, 'model-file')
    print(time.clock() - start)
    print('Finished...')
    
def forward(self, data):
    embeddings = None
    for i in range(len(data)):
        for word in data[i]:
            w = wordEmbedding(self.dictionary, self.chars, word, self.wordEmbeds, self.charEmbeds, self.conv)
            if embeddings is None:
                embeddings = torch.FloatTensor(w)
            else:
                embeddings = torch.cat((embeddings, w))
    probs, v = self.lstm(embeddings.view(len(data[0]), self.batch_size, 30), self.hidden)
    self.hidden = v
    result = self.linear(probs)
    probabilities = []
    for i in range(len(result)):
        probabilities += [[]]
        summation = 0
        for j in range(len(result[i][0])):
            summation += np.exp(result[i][0][j].item())
        for j in range(len(result[i][0])):
            probabilities[i] += [(np.exp(result[i][0][j].item())) / summation]
    modelTags = []
    for i in range(len(data[0])):
        #modelTags += [probabilities[i].index(max(probabilities[i]))]
        modelTags += [max(probabilities[i])]
    return modelTags
    
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
