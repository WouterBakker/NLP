#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# TODO: read brown_vocab_100.txt into word_index_dict
with open('C:/Users/ameer/Desktop/A2_2024/brown_vocab_100.txt', 'r') as file:
    # Read the entire contents of the file such that each line forms its own element in a list
    brown_vocab = file.read().split()

for counts, type_str in enumerate(brown_vocab): # loop over the enumerate object, which always returns the object passed and an index with it
    word_index_dict[type_str.rstrip()] = counts #add newline-stripped word to the dictionary

# TODO: write word_index_dict to word_to_index_100.txt
with open('C:/Users/ameer/Desktop/A2_2024/word_to_index_100.txt', 'w') as file:
    file.write(str(word_index_dict))


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
