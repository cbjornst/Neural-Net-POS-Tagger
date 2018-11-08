# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
import numpy as np
import torch.nn.functional as F

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    tagger = torch.load(model_file)
    data = open(test_file)
    data = data.read().splitlines()
    output = open(out_file, "w")
    tagger.lstm.batch_first = False
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
    t1 = []
    for word in data:
        if word in self.dictionary:
            t1 += [self.dictionary[word]]
        else:
            t1  += [len(self.dictionary) - 1]
    embeddings = self.wordEmbeds(torch.LongTensor(t1))
    #if embeddings is None:
    #    embeddings = w
    #else:
    #    embeddings = torch.cat((embeddings, w))
    self.hidden = (torch.zeros(2, 1, 15), torch.zeros(2, 1, 15))
    probs, self.hidden = self.lstm(embeddings.view(len(data), 1, 15), self.hidden)
    result = self.linear(probs)
    #result = F.softmax(result, dim=2)
    result = torch.argmax(result, dim=2)
    return result

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
