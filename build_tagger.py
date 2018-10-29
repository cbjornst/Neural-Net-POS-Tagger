# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file
    data = open(train_file)
    data = data.read().splitlines()
    dictionary = {}
    chars = {'I': 0, 'n': 1, 'a': 2, 'O': 3, 'c': 4, 't': 5, '.': 6, '1': 7, '9': 8, 'r': 9, 'e': 10, 'v': 11, 'i': 12, 'w': 13, 'o': 14, 'f': 15, '`': 16, 'T': 17, 'h': 18, 'M': 19, 's': 20, 'p': 21, "'": 22, 'C': 23, 'g': 24, 'G': 25, 'd': 26, 'm': 27, '(': 28, 'R': 29, 'l': 30, 'z': 31, 'k': 32, 'S': 33, 'W': 34, 'y': 35, ',': 36, 'L': 37, 'u': 38, '&': 39, 'A': 40, ')': 41, 'b': 42, 'K': 43, 'H': 44, 'E': 45, '-': 46, 'x': 47, 'U': 48, '2': 49, '0': 50, '4': 51, 'B': 52, 'F': 53, 'N': 54, 'D': 55, 'q': 56, '5': 57, '8': 58, '7': 59, '%': 60, '{': 61, '}': 62, 'j': 63, '/': 64, '$': 65, '3': 66, 'Y': 67, ':': 68, 'P': 69, 'Q': 70, 'J': 71, 'V': 72, '?': 73, ';': 74, 'Z': 75, '6': 76, '#': 77, 'X': 78, '!': 79, '\\': 80, '*': 81, '=': 82, '@': 83}
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
    charEmbeds = nn.Embedding(85, 5)
    conv = nn.Conv1d(5, 1, 3)
    lstm = nn.LSTM(15, 15, bidirectional=True)
    tags = nn.Linear(15, 45)
    print('Finished...')
		
def forward(line, wordEmbeds, charEmbeds):
    line = line.split(" ")
    sentence = []
    for word in line:
        word1, word2 = word.rsplit("/", 1)
        word1 = str(word1)
        sentence += [word1]
    sentence = torch.tensor(sentence, dtype=torch.long)
    embeds = wordEmbeds(sentence)
   
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
