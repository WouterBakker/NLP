#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import numpy as np
from generate import GENERATE

### Unigram probabilities
vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict)) #initialize counts to a zero vector

## Calculate unigram probabilities
#iterate through file and update counts
for sent in f:
    sent_list = sent.lower().split()
    for word in sent_list:
        ind = word_index_dict[word]
        counts[ind] += 1

f.close()

#normalize and writeout counts. 

print(counts)
probs = counts / np.sum(counts)
print(f"Probability of \"all\": {probs[0]}")
print(f"Probability of \"resolution\": {probs[-1]}")
      

with open('unigram_probs.txt', 'w') as file:
    file.write(str(probs))

