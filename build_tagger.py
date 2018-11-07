# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

class POSData(Dataset):
    def __init__(self, data, dictionary):
        self.data = data
        self.num = 0
        self.dictionary = dictionary
        self.tags = {"$": 0,"#": 1,"``": 2,"''": 3,"-LRB-": 4,"-RRB-": 5,",": 6,".": 7,":": 8,"CC": 9,"CD": 10,"DT": 11,"EX": 12,"FW": 13,"IN": 14,"JJ": 15,"JJR": 16,"JJS": 17,"LS": 18,"MD": 19,"NN": 20,"NNP": 21,"NNPS": 22,"NNS": 23,"PDT": 24,"POS": 25,"PRP": 26,"PRP$": 27,"RB": 28,"RBR": 29,"RBS": 30,"RP": 31,"SYM": 32,"TO": 33,"UH": 34,"VB": 35,"VBD": 36,"VBG": 37,"VBN": 38,"VBP": 39,"VBZ": 40,"WDT": 41,"WP": 42,"WP$": 43,"WRB": 44}

    def __getitem__(self, index):
        txt = torch.LongTensor(np.zeros(64, dtype=np.int64))
        count = 0
        for word in self.data[index][0]:
            if word in self.dictionary:
                txt[count] = self.dictionary[word]
                count += 1
        label = torch.LongTensor(np.zeros(64, dtype=np.int64))
        for l in self.data[index][1]:
            label[l] = l
        #label = torch.LongTensor([self.data[index][1]])
        return txt, label

    def __len__(self):
        return len(self.data)
    
class Model(nn.Module):
    def __init__(self, V):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(15, 15, bidirectional=True)
        #tagger.lstm = tagger.lstm.cuda()
        self.batch_size = 64
        self.wordEmbeds = nn.Embedding(V, 15)
        self.charEmbeds = nn.Embedding(85, 15)
        self.conv = nn.Conv1d(1, 15, 45)
        self.linear = nn.Linear(30, 45)
        self.softmax = nn.Softmax(2)
        self.hidden = (Variable(torch.zeros(2, self.batch_size, 15)), Variable(torch.zeros(2, self.batch_size, 15)))
    
    def forward(self, data, innerSize):
        #embeddings = None
        #for i in range(len(data)):
        #for word in data[i]:
        #w = self.wordEmbeds(data[0])
        ##w = self.wordEmbedding(self.dictionary, self.chars, word, self.wordEmbeds, self.charEmbeds, self.conv)
        #if embeddings is None:
        #    embeddings = w
        #else:
        #    embeddings = torch.cat((embeddings, w))
        #embeddings = Variable(embeddings)
        #probs, v = self.lstm(embeddings.view(innerSize, self.batch_size, 15), self.hidden)
        probs, v = self.lstm(data, self.hidden)
        self.hidden = v
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(probs, batch_first=True)
        result = self.linear(unpacked)
        #result = self.softmax(result)
        #result = torch.argmax(result, dim=2)
        return result
    
    def wordEmbedding(self, dictionary, chars, word, wEmbeds, cEmbeds, conv):
        if word in dictionary:
            w1 = torch.LongTensor([dictionary[word]])
        else:
            w1 = torch.LongTensor([len(dictionary) - 1])
        e = wEmbeds(w1)
#        ci = None
#        for i in range(len(word)):
#            cj = None
#            for j in range(3):    
#                if (i + j - 1) > 0 and (i + j - 1) < len(word):
#                    c = [chars[word[i + j - 1]]]
#                    c = torch.LongTensor(c)
#                else:
#                    c = [84]
#                    c = torch.LongTensor(c)
#                c = cEmbeds(c)
#                if cj is not None:
#                    cj = torch.cat((cj, c), 1)
#                else:
#                    cj = c
#            cj = cj.unsqueeze(0)
#            c = conv(cj)
#            if ci is not None:
#                ci = torch.cat((ci, c), 2)
#            else:
#                ci = c
#        maximum = nn.MaxPool1d(len(word))
#        w = maximum(ci)
#        w = w.view([1, 15])
        #w = torch.cat((e, w), 1)
        return e
    
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
    chars = {'I': 0, 'n': 1, 'a': 2, 'O': 3, 'c': 4, 't': 5, '.': 6, '1': 7, '9': 8, 'r': 9, 'e': 10, 'v': 11, 'i': 12, 'w': 13, 'o': 14, 'f': 15, '`': 16, 'T': 17, 'h': 18, 'M': 19, 's': 20, 'p': 21, "'": 22, 'C': 23, 'g': 24, 'G': 25, 'd': 26, 'm': 27, '(': 28, 'R': 29, 'l': 30, 'z': 31, 'k': 32, 'S': 33, 'W': 34, 'y': 35, ',': 36, 'L': 37, 'u': 38, '&': 39, 'A': 40, ')': 41, 'b': 42, 'K': 43, 'H': 44, 'E': 45, '-': 46, 'x': 47, 'U': 48, '2': 49, '0': 50, '4': 51, 'B': 52, 'F': 53, 'N': 54, 'D': 55, 'q': 56, '5': 57, '8': 58, '7': 59, '%': 60, '{': 61, '}': 62, 'j': 63, '/': 64, '$': 65, '3': 66, 'Y': 67, ':': 68, 'P': 69, 'Q': 70, 'J': 71, 'V': 72, '?': 73, ';': 74, 'Z': 75, '6': 76, '#': 77, 'X': 78, '!': 79, '\\': 80, '*': 81, '=': 82, '@': 83}
    tags = {"$": 45,"#": 1,"``": 2,"''": 3,"-LRB-": 4,"-RRB-": 5,",": 6,".": 7,":": 8,"CC": 9,"CD": 10,"DT": 11,"EX": 12,"FW": 13,"IN": 14,"JJ": 15,"JJR": 16,"JJS": 17,"LS": 18,"MD": 19,"NN": 20,"NNP": 21,"NNPS": 22,"NNS": 23,"PDT": 24,"POS": 25,"PRP": 26,"PRP$": 27,"RB": 28,"RBR": 29,"RBS": 30,"RP": 31,"SYM": 32,"TO": 33,"UH": 34,"VB": 35,"VBD": 36,"VBG": 37,"VBN": 38,"VBP": 39,"VBZ": 40,"WDT": 41,"WP": 42,"WP$": 43,"WRB": 44}
    V = 0
    trainingData = []
    labels = []
    for line in data:
        line = line.split(" ")
        t1 = []
        t2 = []
        for word in line:
            word1, word2 = word.rsplit("/", 1)
            word1 = str(word1)
            word2 = str(word2)
            if word1 not in dictionary:
                dictionary[word1] = V
                V += 1
            t1 += [V - 1]
            t2 += [tags[word2]]
        trainingData += [torch.LongTensor(t1)]
        labels += [torch.LongTensor(t2)]
    tagger = Model(V + 1)
    #tagger = tagger.cuda()
    tagger.dictionary = dictionary
    tagger.chars = chars
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(tagger.parameters(), lr=learning_rate)
    start = time.clock()
    ln = 0
    batches = []
    i = 0
    while i < len(trainingData):
        if i + 64 > len(trainingData):
            #print(labels[i:])
            batches += [(trainingData[i:], labels[i:])]
        else:
            batches += [(trainingData[i:i+64], labels[i:i+64])]
        i += 64
    #trainingData = POSData(trainingData, dictionary)
    #train_loader = DataLoader(trainingData, batch_size=64, shuffle=True, num_workers=1)
    for epoch in range(1):        
        optimizer = adjust_learning_rate(optimizer, epoch)
        for batch in batches:
            sentence, trainTags = batch
            ordered = sorted(sentence, key=len, reverse=True)
            lengths1 = torch.LongTensor([len(seq) for seq in sentence])
            lengths2 = torch.LongTensor([len(seq) for seq in ordered])
            for i in range(len(ordered)):
                ordered[i] = tagger.wordEmbeds(ordered[i])
            wordBatch = rnn.pad_sequence(ordered, batch_first=True)
            innerSize = len(wordBatch[0])
            wordBatch = rnn.pack_padded_sequence(wordBatch, lengths2, batch_first=True)
            tagger.batch_size = 64
            #wordBatch = rnn.pack_padded_sequence(wordBatch, lengths, batch_first=True)
            ln += 1
            tagger.zero_grad()
            tagger.hidden = (Variable(torch.zeros(2, tagger.batch_size, 15)), Variable(torch.zeros(2, tagger.batch_size, 15)))
            modelTags = tagger.forward(wordBatch, innerSize)
            _, original_idx = lengths1.sort(0)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(modelTags)
            modelTags = modelTags.gather(0, unsorted_idx.long())
            modelTags = torch.squeeze(modelTags)
            loss = lossFunction(modelTags, trainTags)
            loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()
        print(epoch)
    torch.save(tagger, 'model-file')
    print(time.clock() - start)
    print('Finished...')
   
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
