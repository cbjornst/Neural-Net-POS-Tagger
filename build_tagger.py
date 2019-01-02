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
from torch.autograd import Variable
import numpy as np
import random
    
class Model(nn.Module):
    def __init__(self, V):
        #initialize the model
        super(Model, self).__init__()
        self.lstm = nn.LSTM(15, 15, bidirectional=True, batch_first=True)
        self.wordEmbeds = nn.Embedding(V, 15)
        self.batch_size = 64
        self.charEmbeds = nn.Embedding(85, 15)
        self.conv = nn.Conv1d(1, 15, 45)
        self.linear = nn.Linear(30, 45)
        self.hidden = self.init_hidden()
        
    def forward(self, data, innerSize, lengths):
        #convert to word embeddings and pack all padded sequences
        data = self.wordEmbedding(data)
        data = rnn.pack_padded_sequence(data, lengths, batch_first=True)
        self.hidden = self.init_hidden()
        probs, self.hidden = self.lstm(data, self.hidden)
        #unpack in order to view output
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(probs, batch_first=True)
        unpacked = unpacked.contiguous()
        unpacked = unpacked.view(-1, unpacked.shape[2])
        result = self.linear(unpacked)
        result = result.view(self.batch_size, innerSize, 45)
        return result
    
    def init_hidden(self):
        #re-initialize hidden vector as pytorch does not automatically do so
        return (Variable(torch.torch.randn(2, self.batch_size, 15)), Variable(torch.torch.randn(2, self.batch_size, 15))) 
        
    def wordEmbedding(self, words):
        #create word embeddings
        #eventually will also create character embeddings; this is a work in 
        #progress
        e = self.wordEmbeds(words)
        ci = None
        for i in range(len(words)):    
            for j in range(len(words[i])):
                if words[i][j] != 0:
                    word = self.dictionary[words[i][j].item()]
                    cj = None
                    for j in range(3):    
                        if (i + j - 1) > 0 and (i + j - 1) < len(word):
                            c = [self.chars[word[i + j - 1]]]
                            c = torch.LongTensor(c)
                        else:
                            c = [84]
                            c = torch.LongTensor(c)
                        c = self.cEmbeds(c)
                        if cj is not None:
                            cj = torch.cat((cj, c), 1)
                        else:
                            cj = c
                    cj = cj.unsqueeze(0)
                    c = self.conv(cj)
                    if ci is not None:
                        ci = torch.cat((ci, c), 2)
                    else:
                        ci = c
            maximum = nn.MaxPool1d(len(word))
            w = maximum(ci)
            w = w.view([1, 15])
            w = torch.cat((e[i], w), 1)
        return e
    
learning_rate = 1

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train_model(train_file, model_file):
    # train model from train_file
    data = open(train_file)
    data = data.read().splitlines()
    dictionary = {"<PAD>": 0}
    chars = {'I': 0, 'n': 1, 'a': 2, 'O': 3, 'c': 4, 't': 5, '.': 6, '1': 7, '9': 8, 'r': 9, 'e': 10, 'v': 11, 'i': 12, 'w': 13, 'o': 14, 'f': 15, '`': 16, 'T': 17, 'h': 18, 'M': 19, 's': 20, 'p': 21, "'": 22, 'C': 23, 'g': 24, 'G': 25, 'd': 26, 'm': 27, '(': 28, 'R': 29, 'l': 30, 'z': 31, 'k': 32, 'S': 33, 'W': 34, 'y': 35, ',': 36, 'L': 37, 'u': 38, '&': 39, 'A': 40, ')': 41, 'b': 42, 'K': 43, 'H': 44, 'E': 45, '-': 46, 'x': 47, 'U': 48, '2': 49, '0': 50, '4': 51, 'B': 52, 'F': 53, 'N': 54, 'D': 55, 'q': 56, '5': 57, '8': 58, '7': 59, '%': 60, '{': 61, '}': 62, 'j': 63, '/': 64, '$': 65, '3': 66, 'Y': 67, ':': 68, 'P': 69, 'Q': 70, 'J': 71, 'V': 72, '?': 73, ';': 74, 'Z': 75, '6': 76, '#': 77, 'X': 78, '!': 79, '\\': 80, '*': 81, '=': 82, '@': 83}
    tags = {"<PAD>": 45, "$": 0,"#": 1,"``": 2,"''": 3,"-LRB-": 4,"-RRB-": 5,",": 6,".": 7,":": 8,"CC": 9,"CD": 10,"DT": 11,"EX": 12,"FW": 13,"IN": 14,"JJ": 15,"JJR": 16,"JJS": 17,"LS": 18,"MD": 19,"NN": 20,"NNP": 21,"NNPS": 22,"NNS": 23,"PDT": 24,"POS": 25,"PRP": 26,"PRP$": 27,"RB": 28,"RBR": 29,"RBS": 30,"RP": 31,"SYM": 32,"TO": 33,"UH": 34,"VB": 35,"VBD": 36,"VBG": 37,"VBN": 38,"VBP": 39,"VBZ": 40,"WDT": 41,"WP": 42,"WP$": 43,"WRB": 44}
    V = 1
    trainingData = []
    labels = []
    #create dictionary of words
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
            t1 += [dictionary[word1]]
            t2 += [tags[word2]]
        trainingData += [torch.LongTensor(t1)]
        labels += [torch.LongTensor(t2)]
    tagger = Model(V + 1)
    #tagger = tagger.cuda()
    tagger.dictionary = dictionary
    tagger.chars = chars
    #use cross entropy loss and SGD for the neural net
    optimizer = optim.SGD(tagger.parameters(), lr=learning_rate)
    lossFunction = nn.CrossEntropyLoss(ignore_index=45)
    start = time.clock()
    batches = []
    i = 0
    while i < len(trainingData):
        if i + 64 > len(trainingData):
            batches += [(trainingData[i:], labels[i:])]
        else:
            batches += [(trainingData[i:i+64], labels[i:i+64])]
        i += 64
    train_loss_ = []
    for epoch in range(5):        
        #manually shuffle and pass batches in as the built-in pytorch function
        #was having problems
        optimizer = adjust_learning_rate(optimizer, epoch)        
        total_loss = 0.0
        total = 0.0
        random.shuffle(batches)
        for batch in batches:
            #pad each sentence and tag list to ensure they are the same length
            sentence, trainTags = batch
            ordered = sorted(sentence, key=len, reverse=True)
            ordered2 = sorted(trainTags, key=len, reverse=True)
            lengths = torch.LongTensor([len(seq) for seq in ordered])
            wordBatch = rnn.pad_sequence(ordered, batch_first=True)
            trainTags = rnn.pad_sequence(ordered2, batch_first=True)
            innerSize = len(wordBatch[0])
            tagger.batch_size = len(sentence)
            #zero gradients as pytorch does not automatically do this
            tagger.zero_grad()
            modelTags = tagger.forward(wordBatch, innerSize, lengths)
            trainTags = trainTags.view(-1)
            modelTags = modelTags.view(-1, 45)
            loss = lossFunction(modelTags, trainTags)
            loss.backward()
            optimizer.step()
            total += len(trainTags)
            total_loss += loss.item()
        #rudimentary loss calculation, could use improvement
        train_loss_.append(total_loss * 1000 / total)
        print(train_loss_[-1])
    torch.save(tagger, model_file)
    print(time.clock() - start)
    print('Finished...')
   
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
