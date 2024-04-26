import numpy as np
from sklearn.preprocessing import normalize
import codecs



### Unigram model probabilities

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #build dictionary with word indices
    word_index_dict[line.rstrip()] = i


f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict)) #initialize counts to a zero vector

#iterate through file and update counts
for sent in f:
    sentence = sent.lower().split()
    for word in sentence:
        ind = word_index_dict[word]
        counts[ind] += 1

f.close()

# calculate unigram word probabilities
probs_unigram = counts / np.sum(counts)


### Bigram model probabilities

counts = np.zeros((len(word_index_dict), len(word_index_dict))) #initialize numpy 0s matrix

#iterate through file and update counts
f = codecs.open("brown_100.txt")

prev_word = "<s>"

for sent in f:
    words = sent.lower().split()
    for word in words:
        next_word = word
        ind_y = word_index_dict[next_word]
        ind_x = word_index_dict[prev_word]
        counts[ind_x, ind_y] += 1
        prev_word = next_word

# normalize counts
probs_bigram_unsmoothed = normalize(counts, norm='l1', axis=1)

counts += 0.1
probs_bigram_smoothed = normalize(counts, norm='l1', axis=1)


f.close()



#### Calculating sentence probabilities and perplexities

toy_corpus_file = open("toy_corpus.txt", "r")
toy_corpus_text = toy_corpus_file.read()
sentences = toy_corpus_text.split('\n')

sentlen = []
sentprob = []




# Unigram probability
for sentence in sentences:
    sent = sentence.lower().split()
    prob = np.prod([probs_unigram[word_index_dict[x]] for x in sent])
    sentlen.append(len(sent))
    sentprob.append(prob)


perplexity_unigram = [1/(pow(prob, 1.0/length)) for prob, length in zip(sentprob, sentlen)]
perplexity_unigram



# Bigram probability unsmoothed
sentlen = []
sentprob = []


for sentence in sentences:
    sent = sentence.lower().split()
    prob = np.prod([probs_bigram_unsmoothed[word_index_dict[x],word_index_dict[y]] for x,y in zip(sent[:-1], sent[1:])])
    sentlen.append(len(sent)-1)
    sentprob.append(prob)


perplexity_bigram = [1/(pow(prob, 1.0/length)) for prob, length in zip(sentprob, sentlen)]
perplexity_bigram





# Bigram probability smoothed
sentlen = []
sentprob = []


for sentence in sentences:
    sent = sentence.lower().split()

    wordprobs = [probs_bigram_smoothed[word_index_dict[x],word_index_dict[y]] for x,y in zip(sent[:-1], sent[1:])]
    prob = np.prod(wordprobs)
    sentlen.append(len(sent)-1)
    sentprob.append(prob)


perplexity_bigram_smoothed = [1/(pow(prob, 1.0/length)) for prob, length in zip(sentprob, sentlen)]
perplexity_bigram_smoothed



[[x,y] for x,y in zip(sent[:-1], sent[1:])]






