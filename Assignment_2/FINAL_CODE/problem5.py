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
## e.g., in: the: past: 0
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
            # removes word to ensure length of wordlist is always 2 (last two words)
            wordlist = wordlist[1:]
        if wordlist in ps_two:
            hit = True


f.close()



list_probs = []

for p in ps_three:    
    #count of target value, the random variable we want to obtain the likelihood for, e.g. "past" in P(past|in, the)
    target_value = dd[p[0]][p[1]][p[2]]
    # Obtain the conditioning variable, e.g. "in, the" in P(past|in, the)
    conditioning_variable = dd[p[0]][p[1]]
    #total count for the values in ps_two, the conditioning variable
    values = conditioning_variable.values()
    p_unsmoothed = target_value / sum(values)
    ## To obtain smoothed count:
    ### There are 813 words in the corpus, and every word should get +0.1
    ### So, we need to obtain the complement of words that are not in p(x|y,z): 813 - #words_in_p(x|y,z)
    ### All counts for those words are 0, so complement*0.1 gives the right value to calculate smoothed probability
    smoothed_values = [x + 0.1 for x in conditioning_variable.values()] # obtain smoothed counts by adding 0.1 to the values
    complement_smoothed_values = 0.1*(813 - len(smoothed_values))
    p_smoothed = (target_value+0.1) / (sum(smoothed_values) + complement_smoothed_values)

    list_probs.append([p_unsmoothed, p_smoothed])


print("Unsmoothed value | smoothed value")
for p in list_probs:
    print(f"{p[0]} | {p[1]}")


