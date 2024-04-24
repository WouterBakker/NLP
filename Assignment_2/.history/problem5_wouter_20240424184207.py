import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs
from collections import defaultdict 


p1 = ["in", "the"]
p2 = ["in", "the"]
p3 = ["the", "jury"]
p4 = ["the", "jury"]
p5 = ["jury", "said"]
p6 = ["agriculture", "teacher"]
# Saves the two words we condition on (e.g., p(past|in, the)) to a list 
ps_two = [p1, p2, p3, p4, p5, p6]

# Saves the full trigram conditionals to a list
p1 = ["in", "the", "past"]
p2 = ["in", "the", "time"]
p3 = ["the", "jury", "said"]
p4 = ["the", "jury", "recommended"]
p5 = ["jury", "said", "that"]
p6 = ["agriculture", "teacher", ","]

ps_three = [p1, p2, p3, p4, p5, p6]

# Initializes a defaultdict of structure:
## Word1: Word2: Word3: count
## e.g., in: the: past: 1
dd = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


f = codecs.open("brown_100.txt")


wordlist = []
hit = False

for sent in f:
    # Proper formatting of sentences in f
    sent_list = sent.lower().split()
    for word in sent_list:
        #Keep track of which words were the previous 2 words
        wordlist.append(word)
        #hit = True, if any of the two-word combinations in p_two are found (only ones relevant for the trigram probability)
        #in which case, adds count to the defaultdict
        if hit:
            dd[wordlist[0]][wordlist[1]][wordlist[2]] += 1
        if len(wordlist) == 3:
            # removes word to ensure length of wordlist is always 2
            wordlist = wordlist[1:]
        if wordlist in ps_two:
            hit = True


f.close()

# dd["in"]["the"]["past"] / sum(dd["in"]["the"].values())
# dd["in"]["the"]["time"] / sum(dd["in"]["the"].values())
# dd["the"]["jury"]["said"] / sum(dd["in"]["the"].values())
# dd["the"]["jury"]["recommended"] / sum(dd["in"]["the"].values())
# dd["jury"]["said"]["that"] / sum(dd["in"]["the"].values())
# dd["agriculture"]["teacher"][","] / sum(dd["in"]["the"].values())


list_probs = []

for p in ps_three:
    #total count for the values in ps_two, the conditioning variable
    values = dd[p[0]][p[1]].values()
    #count of target value, the random variable we want to obtain the likelihood for
    target_value = dd[p[0]][p[1]][p[2]]
    smoothed_values = [x + 0.1 for x in dd["in"]["the"].values()]
    p_unsmoothed = target_value / sum(values)
    p_smoothed = target_value / sum(smoothed_values)

    list_probs.append([p_unsmoothed, p_smoothed])


print("smoothed value | unsmoothed value")
for p in list_probs:
    print(f"{p[0]} | {p[1]}")
    
    
    
