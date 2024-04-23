#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


with open('C:/Users/ameer/Desktop/A2_2024/brown_vocab_100.txt', 'r') as file:
    # Read the entire contents of the file such that each line forms its own element in a list
    vocab = file.read().split()

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i 

with open("C:/Users/ameer/Desktop/A2_2024/brown_100.txt") as file:
    f = file.readlines()

counts = np.zeros(len(vocab)) #TODO: initialize counts to a zero vector


#TODO: iterate through file and update counts
for i, sent in enumerate(f):
    stripping = sent.replace("<s>","").replace("</s>", "").rstrip().lower().split() #go through each sentence in the file, remove <s> and </s>, strip the whitespace, convert to lowercase and split the sentence to individual words
    for words in stripping: #go through each word of each sentence
        counts[list(word_index_dict.keys()).index(words)] += 1 #update the number of words mentioned in the given file
    

#TODO: normalize and writeout counts. 
probs = counts / np.sum(counts) #directly taken from the assignment file


with open('C:/Users/ameer/Desktop/A2_2024/unigram_probs.txt', 'w') as file:
    file.write(str(probs))