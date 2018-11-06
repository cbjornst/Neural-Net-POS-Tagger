# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    tags = ["$","#","``","''","-LRB-","-RRB-",",",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
    lstm = torch.load(model_file)
    data = open(test_file)
    data = data.read().splitlines()
    output = open(out_file, "w")
    for line in data:
        out = ""
        line = line.split(" ")
        result = lstm.forward2(lstm, line)
        for i in range(len(line)): 
            out += line[i] + "/" + tags[result[i]] + " "
        output.write(out + "\n")
        print(out)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
