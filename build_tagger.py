# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import random
    
class Model(nn.Module):
    def __init__(self, V):
        #initialize the model
        super(Model, self).__init__()
        self.batch_size = 64
        self.hSize = 20
        self.numTags = 45
        self.lstm = nn.LSTM(self.hSize, self.hSize, bidirectional=True, batch_first=True)
        self.wordEmbeds = nn.Embedding(V, self.hSize)
        self.charEmbeds = nn.Embedding(85, self.hSize)
        self.conv = nn.Conv1d(1, self.hSize, self.numTags)
        self.linear = nn.Linear(self.hSize * 2, self.numTags)
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
        result = result.view(self.batch_size, innerSize, self.numTags)
        return result
    
    def init_hidden(self):
        #re-initialize hidden vector as pytorch does not automatically do so
        return (Variable(torch.torch.randn(2, self.batch_size, self.hSize)), Variable(torch.torch.randn(2, self.batch_size, self.hSize))) 
       
    def charEmbedding(self, word):
        wordLen = len(word)
        wordEmbed = None
        index = 0
        for letter in range(1, wordLen):
            letters = self.dictionary[word[letter].item()]
            charEmbed = None
            for j in range(-1, 2):    
                if (letter + j) > 0 and (letter + j) < wordLen:
                    c = [self.chars[word[letter + j]]]
                    c = torch.LongTensor(c)
                else:
                    c = [84]
                    c = torch.LongTensor(c)
            c = self.cEmbeds(c)
            if charEmbed is not None:
                charEmbed = torch.cat((charEmbed, c), 1)
            else:
                charEmbed = c
        charEmbed = charEmbed.unsqueeze(0)
        c = self.conv(charEmbed)
        if wordEmbed is not None:
            wordEmbed = torch.cat((wordEmbed, c), 2)
        else:
            wordEmbed = c
        maximum = nn.MaxPool1d(len(letters))
        w = maximum(wordEmbed)
        w = w.view([1, self.hSize])
        #w = torch.cat((embedding[index], w), 1)
        index += 1
        
    def wordEmbedding(self, words):
        #create word embeddings
        embedding = self.wordEmbeds(words)
        return embedding
    
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
    tagList = {"<PAD>": 45, "$": 0,"#": 1,"``": 2,"''": 3,"-LRB-": 4,"-RRB-": 5,",": 6,".": 7,":": 8,"CC": 9,"CD": 10,"DT": 11,"EX": 12,"FW": 13,"IN": 14,"JJ": 15,"JJR": 16,"JJS": 17,"LS": 18,"MD": 19,"NN": 20,"NNP": 21,"NNPS": 22,"NNS": 23,"PDT": 24,"POS": 25,"PRP": 26,"PRP$": 27,"RB": 28,"RBR": 29,"RBS": 30,"RP": 31,"SYM": 32,"TO": 33,"UH": 34,"VB": 35,"VBD": 36,"VBG": 37,"VBN": 38,"VBP": 39,"VBZ": 40,"WDT": 41,"WP": 42,"WP$": 43,"WRB": 44}
    V = 1
    trainingData = []
    labels = []
    #create dictionary of words
    for line in data:
        line = line.split(" ")
        words = []
        tags = []
        for taggedWord in line:
            word, tag = taggedWord.rsplit("/", 1)
            word = str(word)
            tag = str(tag)
            if word not in dictionary:
                dictionary[word] = V
                V += 1
            words += [dictionary[word]]
            tags += [tagList[tag]]
        trainingData += [torch.LongTensor(words)]
        labels += [torch.LongTensor(tags)]
    tagger = Model(V + 1)
    tagger.dictionary = dictionary
    tagger.chars = chars
    #use cross entropy loss and SGD for the neural net
    optimizer = optim.SGD(tagger.parameters(), lr=learning_rate)
    lossFunction = nn.CrossEntropyLoss(ignore_index=45)
    start = time.clock()
    batches = []
    i = 0
    tdSize = len(trainingData)
    while i < tdSize:
        if i + 64 > tdSize:
            batches += [(trainingData[i:], labels[i:])]
        else:
            batches += [(trainingData[i:i+64], labels[i:i+64])]
        i += 64
    train_loss_ = []
    for epoch in range(10):        
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
