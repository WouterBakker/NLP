#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# read brown_vocab_100.txt into word_index_dict
vocab = open("brown_vocab_100.txt")

for num, x in enumerate(vocab):
    word_index_dict[x.rstrip()] = num


# write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as file:
    for key, value in word_index_dict.items():
        file.write(f'{key}: {value}\n')


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
