#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
import codecs

### Unsmoothed bigram probabilities
vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i


f = codecs.open("brown_100.txt")

counts = np.zeros((len(word_index_dict), len(word_index_dict))) #initialize numpy 0s matrix

#iterate through file and update counts

prev_word = "<s>"
i = 0 #ensures that first word is skipped
for sent in f:
    sent_list = sent.lower().split()
    for word in sent_list:
        if i == 0:
            i+=1
            continue
        next_word = word
        ind_y = word_index_dict[next_word]
        ind_x = word_index_dict[prev_word]
        counts[ind_x, ind_y] += 1
        prev_word = next_word
        

# normalize counts
probs = normalize(counts, norm='l1', axis=1)


#writeout bigram probabilities

the_ind = word_index_dict["the"]
all_ind = word_index_dict["all"]
jury_ind = word_index_dict["jury"]
campaign_ind = word_index_dict["campaign"]
calls_ind = word_index_dict["calls"]
anonymous_ind = word_index_dict["anonymous"]

fourprobs = [probs[all_ind, the_ind],
                probs[the_ind, jury_ind],
                probs[the_ind, campaign_ind],
                probs[anonymous_ind, calls_ind]]

print(f"p(the|all): {fourprobs[0]}")
print(f"p(jury|the): {fourprobs[1]}")



with open("bigram_probs.txt", 'w') as file:
    for p in fourprobs:
        file.write(f"{p}\n")


f.close()