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
        self.num = 0

    def __getitem__(self, index):
        txt = self.data[index][0]
        label = self.data[index][1]
        return txt, label

    def __len__(self):
        return len(self.data)
    
class Model(nn.Module):
    def __init__(self, V):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(30, 30, bidirectional=True)
        #tagger.lstm = tagger.lstm.cuda()
        self.batch_size = 1
        self.wordEmbeds = nn.Embedding(V, 15)
        self.charEmbeds = nn.Embedding(85, 15)
        self.conv = nn.Conv1d(1, 15, 45)
        self.linear = nn.Linear(60, 45)
        self.softmax = nn.Softmax(2)
        self.hidden = (Variable(torch.zeros(2, self.batch_size, 30)), Variable(torch.zeros(2, self.batch_size, 30)))
    
    def forward(self, data, innerSize):
        embeddings = None
        for i in range(len(data)):
            for word in data[i]:
                w = self.wordEmbedding(self.dictionary, self.chars, word, self.wordEmbeds, self.charEmbeds, self.conv)
                if embeddings is None:
                    embeddings = w
                else:
                    embeddings = torch.cat((embeddings, w))
        probs, v = self.lstm(embeddings.view(self.batch_size, innerSize, 30), self.hidden)
        self.hidden = v
        result = self.linear(probs)
        result = self.softmax(result)
        #probabilities = []
        #for i in range(len(result)):
        #    probabilities += [[]]
        #    for j in range(len(result[i])):
        #        probabilities[i] += [[]]
        #        summation = 0
        #        for k in range(len(result[i][j])):
        #            summation += np.exp(result[i][j][k].item())
        #        for k in range(len(result[i][j])):
        #            probabilities[i][j] += [(np.exp(result[i][j][k].item())) / summation]
        modelTags = []
        for i in range(len(data)):
            modelTags += [[[]]]
            for j in range(len(data[i])):
                #modelTags += [probabilities[i].index(max(probabilities[i]))]
                modelTags[i][0] += [max(result[i][j]).item()]
        return modelTags
    
    def wordEmbedding(self, dictionary, chars, word, wEmbeds, cEmbeds, conv):
        if word in dictionary:
            w1 = torch.LongTensor([dictionary[word]])
        else:
            w1 = torch.LongTensor([len(dictionary) - 1])
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
    
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

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
        trainingData += [[t1, t2]]
    tagger = Model(V + 1)
    #tagger = tagger.cuda()
    tagger.forward2 = forward2
    tagger.dictionary = dictionary
    tagger.chars = chars
    s = nn.Sigmoid()
    lossFunction = nn.BCELoss()
    optimizer = optim.SGD(tagger.parameters(), lr=learning_rate)
    start = time.clock()
    ln = 0
    trainingData = Dataset(trainingData)
    train_loader = DataLoader(trainingData, batch_size=64, shuffle=True)
    for epoch in range(50):        
        optimizer = adjust_learning_rate(optimizer, epoch)
        for sentence, trainTags in train_loader:
            innerSize = len(sentence[0])
            correctTags = []
            for i in range(len(trainTags)):
                t = []
                for j in range(len(trainTags[i])):
                    t += [tags.index(trainTags[i][j])]
                correctTags += [[t]]
            correctTags = s(torch.FloatTensor(np.asarray(correctTags)))
            tagger.batch_size = len(sentence)
            ln += 1
            if ln % 50 == 0:
                print(time.clock() - start)
            tagger.zero_grad()
            tagger.hidden = (Variable(torch.zeros(2, innerSize, 30)), Variable(torch.zeros(2, innerSize, 30)))
            modelTags = tagger.forward(sentence, innerSize)
            modelTags = torch.FloatTensor(np.asarray(modelTags))
            loss = lossFunction(modelTags, correctTags)
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()
    torch.save(tagger, 'model-file')
    print(time.clock() - start)
    print('Finished...')

def forward2(self, data):
    embeddings = None
    for word in data:
        w = self.wordEmbedding(self.dictionary, self.chars, word, self.wordEmbeds, self.charEmbeds, self.conv)
        if embeddings is None:
            embeddings = w
        else:
            embeddings = torch.cat((embeddings, w))
    self.hidden = (torch.zeros(2, 1, 30), torch.zeros(2, 1, 30))
    probs, v = self.lstm(embeddings.view(len(data), 1, 30), self.hidden)
    self.hidden = v
    result = self.linear(probs)
    probabilities = []
    for i in range(len(result)):
        probabilities += [[]]
        summation = 0
        for k in range(len(result[i][0])):
            summation += np.exp(result[i][0][k].item())
        for k in range(len(result[i][0])):
            probabilities[i] += [(np.exp(result[i][0][k].item())) / summation]
    modelTags = []
    for i in range(len(data)):
       modelTags += [probabilities[i].index(max(probabilities[i]))]
    return modelTags

   
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
