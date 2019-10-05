# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    tagger = torch.load(model_file, map_location='cpu')
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
    self.hidden = (torch.zeros(2, 1, 20), torch.zeros(2, 1, 20))
    for word in data:
        if word in self.dictionary:
            t1 += [self.dictionary[word]]
        else:
            t1  += [len(self.dictionary) - 1]
    t1 = torch.LongTensor(t1)
    embeddings = self.wordEmbeds(t1)
    probs, self.hidden = self.lstm(embeddings.view(len(data), 1, 20), self.hidden)
    result = self.linear(probs)
    result = torch.argmax(result, dim=2)
    return result

class Model(nn.Module):
    def __init__(self, V):
        #initialize the model
        super(Model, self).__init__()
        self.batch_size = 64
        self.hSize = 20
        self.numTags = 45
        self.lstm = nn.LSTM(self.hSize, self.hSize, bidirectional=True, batch_first=True)
        self.wordEmbeds = nn.Embedding(V, self.hSize)
        #self.charEmbeds = nn.Embedding(85, self.hSize)
        #self.conv = nn.Conv1d(1, self.hSize, self.numTags)
        self.linear = nn.Linear(self.hSize * 2, self.numTags)
        self.hidden = self.init_hidden()
        
    def forward(self, data, innerSize, lengths):
        data = self.wordEmbeds(data)
        data = rnn.pack_padded_sequence(data, lengths, batch_first=True)
        self.hidden = self.init_hidden()
        probs, self.hidden = self.lstm(data, self.hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(probs, batch_first=True)
        unpacked = unpacked.contiguous()
        unpacked = unpacked.view(-1, unpacked.shape[2])
        result = self.linear(unpacked)
        result = result.view(self.batch_size, innerSize, self.numTags)
        return result
    
    def init_hidden(self):
        return (Variable(torch.torch.randn(2, self.batch_size, self.hSize)), Variable(torch.torch.randn(2, self.batch_size, self.hSize))) 
        
    def wordEmbedding(self, dictionary, chars, words, wEmbeds, cEmbeds, conv):
        w1 = []
        for word in words:  
            if word in dictionary:
                w1 += [word]
            else:
                w1  += [len(dictionary) - 1]
        e = wEmbeds(torch.LongTensor(w1))
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

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
