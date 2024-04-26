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
    # Creates new entry for each word in the corpus in the order in which they appear
    word_index_dict[x.rstrip()] = num


# Write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as file:
    for key, value in word_index_dict.items():
        file.write(f'{key}: {value}\n')

file.close()


print(f"Word index for \"all\": {word_index_dict['all']}")
print(f"Word index for \"resolution\": {word_index_dict['resolution']}")

