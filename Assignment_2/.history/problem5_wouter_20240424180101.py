import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs
from collections import defaultdict 



f = codecs.open("brown_100.txt")

wordlist = []
count = 0

for sent in f:
    sent_list = sent.lower().split()
    for word in sent_list:
        wordlist.append(word)
        if len(wordlist) > 2:
            wordlist = wordlist[1:]
        
        if wordlist == ["in", "the"]:
            count += 1




p1 = ["in", "the"]
p2 = ["in", "the"]
p3 = ["the", "jury"]
p4 = ["the", "jury"]
p5 = ["jury", "said"]
p6 = ["agriculture", "teacher"]

ps_two = [p1, p2, p3, p4, p5, p6]

p1 = ["in", "the", "past"]
p2 = ["in", "the", "time"]
p3 = ["the", "jury", "said"]
p4 = ["the", "jury", "recommended"]
p5 = ["jury", "said", "that"]
p6 = ["agriculture", "teacher", ","]

ps_three = [p1, p2, p3, p4, p5, p6]

dd = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


wordlist = []
hit = False

for sent in f:
    sent_list = sent.lower().split()
    for word in sent_list:
        wordlist.append(word)
        
        if hit:
            dd[wordlist[0]][wordlist[1]][wordlist[2]] += 1
        if len(wordlist) == 3:
            wordlist = wordlist[1:]
        if wordlist in ps_two:
            hit = True


dd["in"]["the"]["past"] / sum(dd["in"]["the"].values())
dd["in"]["the"]["time"] / sum(dd["in"]["the"].values())
dd["the"]["jury"]["said"] / sum(dd["in"]["the"].values())
dd["the"]["jury"]["recommended"] / sum(dd["in"]["the"].values())
dd["jury"]["said"]["that"] / sum(dd["in"]["the"].values())
dd["agriculture"]["teacher"][","] / sum(dd["in"]["the"].values())


list_probs = []


for p in ps_three:
    values = dd[p[0]][p[1]].values()
    target_value = dd[p[0]][p[1]][p[2]]
    smoothed_values = [x + 0.1 for x in dd["in"]["the"].values()]
    p_unsmoothed = target_value / sum(values)
    p_smoothed = target_value / sum(smoothed_values)

    list_probs.append([p_unsmoothed, p_smoothed])


print("smoothed value | unsmoothed value")
for p in list_probs:
    print(f"{p[0]} | {p[1]}")