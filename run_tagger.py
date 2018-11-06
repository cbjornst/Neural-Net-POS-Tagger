# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
import numpy as np
import torch.nn as nn

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    tagger = torch.load(model_file)
    data = open(test_file)
    data = data.read().splitlines()
    output = open(out_file, "w")
    for line in data:
        out = ""
        line = line.split(" ")
        result = forward2(tagger, line)
        for i in range(len(line)): 
            out += line[i] + "/" + tags[result[i]] + " "
        output.write(out + "\n")
    print('Finished...')

def forward2(self, data):
    embeddings = None
    for word in data:
        w = wordEmbedding(self.dictionary, self.chars, word, self.wordEmbeds, self.charEmbeds, self.conv)
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

def wordEmbedding(dictionary, chars, word, wEmbeds, cEmbeds, conv):
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

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
