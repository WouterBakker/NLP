import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter, OrderedDict
import numpy as np



### Compute a list of unique words sorted by descending frequency for entire corpus
# Counts of each words
words = [x.lower() for x in brown.words()]#list(brown.words())
N = len(words)
wordcounts = Counter(words)
# most frequent words
sorted_wordcounts = dict(wordcounts.most_common())

# calculate unigram probabilities
counts_unigram = sum(wordcounts.values())
probs_unigram = {x:(y/counts_unigram) for x,y in sorted_wordcounts.items() if y > 10}
# List of included words
included_words = probs_unigram.keys()

# N = len(included_words)

counts_bigram = defaultdict(lambda: defaultdict(int))
probs_bigram = defaultdict(lambda: defaultdict(int))


# {x:y for x,y in probs_unigram.items() if y==0}


for x,y in zip(words[:-1], words[1:]):
    counts_bigram[x][y] += 1

for key, inner_dict in counts_bigram.items():
    total_count = sum(inner_dict.values())
    for inner_key in inner_dict.keys():
        probs_bigram[key][inner_key] = inner_dict[inner_key] / total_count



pmis = {}

cw1w2, cw1, cw2 = [], [], []

p = []

for x,y in zip(words[:-1], words[1:]):
    if x in included_words and y in included_words: #only calculated probability if word occurs more than 10 times in the corpus
        p_x = wordcounts[x]
        p_y = wordcounts[y]        
        p_xy = counts_bigram[x][y]

        cw1.append(p_x)
        cw2.append(p_y)
        cw1w2.append(p_xy)
        pmis[f"[{x}, {y}]"] = np.log((p_xy*N)/(p_x*p_y))



# pmis = {}

# cw1w2, cw1, cw2 = [], [], []

# p = []

# for x,y in zip(words[:-1], words[1:]):
#     if x in included_words and y in included_words: #only calculated probability if word occurs more than 10 times in the corpus
#         p_x = probs_unigram[x]
#         p_y = probs_unigram[y]        
#         p_xy = probs_bigram[x][y]

#         cw1.append(p_x)
#         cw2.append(p_y)
#         cw1w2.append(p_xy)
#         # pmis[f"[{x}, {y}]"] = np.log((p_xy*N)/(p_x*p_y))
#         pmis[f"[{x}, {y}]"] = np.log((p_xy)/(p_x*p_y))




sorted_pmis = sorted(pmis.items(), key=lambda x: x[1], reverse = True)

# 20 most common items
most_common = sorted_pmis[:20]
most_common_rounded = [(x, np.round(y, 2)) for x,y in most_common]

# 20 least common items
least_common = sorted_pmis[-20:]
least_common_rounded = [(x, np.round(y, 2)) for x,y in least_common]
