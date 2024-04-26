import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

#### Code for problem7

##### Word indices

vocab = codecs.open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #import part 1 code to build dictionary
    word_index_dict[line.rstrip()] = i


##### Unigram model

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict)) #initialize counts to a zero vector

#iterate through file and update counts
for sent in f:
    sent_list = sent.lower().split()

    for word in sent_list:
        ind = word_index_dict[word]
        counts[ind] += 1

f.close()

#normalize counts. 
probs_unigram = counts / np.sum(counts)


##### Bigram model

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
        

# calculate unsmoothed bigram probabilities
probs_bigram = normalize(counts, norm='l1', axis=1)

# calculate smoothed bigram probabilities
counts += 0.1
probs_bigram_smoothed = normalize(counts, norm='l1', axis=1)


#open and read the toy_corpus.txt
toy_corpus_file = open("toy_corpus.txt", "r")
toy_corpus_text = toy_corpus_file.read()
toy_corpus_file.close()
toy_corpus_sent = toy_corpus_text.split('\n')


#### Calculating sentence probabilities and perplexities


## Calculating sentence probability and perplexity for the unigram model


#open to write the probabilities into the new txt file
output_file = open("unigram_eval.txt", "w")

for sent in toy_corpus_sent:
    sent_list = sent.lower().split()
    sentprob = 1
    for word in sent_list:
        ind = word_index_dict[word]
        wordprob = probs_unigram[ind]
        sentprob *= wordprob  #sentence probability
    
    #calculates sentence length
    sent_len = len(sent_list)
    
    # Calculate perplexity
    perplexity = 1 / (pow(sentprob, 1.0 / sent_len))
    #write perplexity to file
    output_file.write(str(perplexity) + '\n')


output_file.close()

print('sent_len: ', sent_len)
print('sentprob: ', sentprob)
#cheeck perplexity of 2nd sentence is 153
print('perplexity of the 2nd sentence: ', perplexity)


## Calculating sentence probability and perplexity for the unsmoothed bigram model

#start and open the perplexity file
output_file = open("bigram_eval.txt", "w")

#perplexities calculations for toy corpus
#iterates through sentences

for sent in toy_corpus_sent:
    sent_list = sent.lower().split()
    sentprob = 1
    # initialize variables to ensure the first word of each sentence is skipped
    prev_word = "<s>"
    i = 0   
    for word in sent_list:
        if i == 0:
            i += 1
            continue
        next_word = word
        ind_y = word_index_dict[next_word]
        ind_x = word_index_dict[prev_word]
        wordprob = probs_bigram[ind_x, ind_y]
        sentprob *= wordprob
        prev_word = next_word
    # nr. of bigrams = nr. of tokens - 1
    sent_len = len(sent_list) - 1
    #calculate perplexity
    perplexity = 1/(pow(sentprob, (1.0/sent_len)))  
    output_file.write(str(perplexity) + '\n')

#close file
output_file.close()

print('sent_len: ', sent_len)
print('sentprob: ', sentprob)
#check perplexity of 2nd sentence is about 7.57 for the MLE bigram
print('perplexity of the 2nd sentence for MLE bigram is: ', perplexity)



## Calculating sentence probability and perplexity for the smoothed bigram model

#start and open the perplexity file
output_file = open("smoothed_eval.txt", "w")

#perplexities calculations for toy corpus
#iterates through sentences

for sent in toy_corpus_sent:
    sent_list = sent.lower().split()
    sentprob = 1
    # initialize variables to ensure the first word of each sentence is skipped
    ## In this case, we do this at the start of each sentence, so within the 1st for loop
    prev_word = "<s>"
    i = 0
    for word in sent_list:
        if i == 0:
            i += 1
            continue
        next_word = word
        ind_y = word_index_dict[next_word]
        ind_x = word_index_dict[prev_word]
        wordprob = probs_bigram_smoothed[ind_x, ind_y]
        sentprob *= wordprob
        prev_word = next_word
    sent_len = len(sent_list) - 1
    #calculate perplexity
    perplexity = 1/(pow(sentprob, (1.0/sent_len)))  
    output_file.write(str(perplexity) + '\n')

#close file
output_file.close()

print('sent_len: ', sent_len)
print('sentprob: ', sentprob)
#check perplexity of 2nd sentence is about 54.28 for the MLE smoothed bigram
print('perplexity of the 2nd sentence for MLE bigram is: ', perplexity)