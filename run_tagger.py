# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    result = [1, 2, 3]
    probabilities = []
    for i in range(len(result)):
        probabilities += [[]]
        summation = 0
        for j in range(len(result[i][0])):
            summation += np.exp(result[i][0][j].item())
        for j in range(len(result[i][0])):
            probabilities[i] += [(np.exp(result[i][0][j].item())) / summation]
    tag = []
    for i in range(len(line)):
        tag += [tags[probabilities[i].index(max(probabilities[i]))]]
    print(tag)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
