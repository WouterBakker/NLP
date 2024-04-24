#sasha problem 0, assignment 2

# import the brown corpus from the nltk package
import matplotlib.pyplot as plt
from nltk.corpus import brown
from collections import Counter
from nltk import pos_tag

# (i)
#Compute a list of unique words sorted by descending frequency for (i) the whole corpus, while avoiding punctuation
entire_corpus = [word for word in brown.words() if word.isalpha()]

#computes the frequency of each word in brown corpus
freq_words = {}
for word in entire_corpus:
    #if a word was not yet encountered, count is set at 1
    #if word is already encountered, counter adds +1
    freq_words[word] = freq_words.get(word, 0) + 1

#looks at all the items in freq_words, then lmbda func selects the frequency part of the tuple to sort
#the reverse=True requests it sorts in descending order (so most frequent words at top of list)
sorted_words = sorted(freq_words.items(), key=lambda x: x[1], reverse=True)
print('The top unique words sorted by descending freq: ')
print(sorted_words[:15])

# (ii)
# Compute a list of unique words sorted by descending frequency (ii) for two different genres of your choice.
#check what 'genres' (categories) exist in the corpus
#print(brown.categories())

# answer: ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government',
# 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews',
# 'romance', 'science_fiction']

#pick genres together
genres = ['humor', 'mystery']

#iterate over both generes to collect all the unique words into one corpus
both_genre_corpus =[]
for genre in genres:
    both_genre_corpus.extend([word for word in brown.words(categories=genre) if word.isalpha()])

#go thru adding # of times words appear in both genres, same as before, and sort
freq_words_both = {}
for word in both_genre_corpus:
    freq_words_both[word] = freq_words_both.get(word, 0) + 1

sorted_both_words = sorted(freq_words_both.items(), key=lambda x: x[1], reverse=True)

print(f'top fifteen unique words for humor and mystery: ')
print(sorted_both_words[:15])

############################################
# Next looking at whole Brown corpus
# number of tokens, includes punctuation
number_tokens= len(brown.words())
print('Number of tokens: ', number_tokens)

# token types
number_token_types = len(set(brown.words()))
print('Number of token types: ', number_token_types)

#number of words, excluding punctuation
number_words = len([word for word in brown.words() if word.isalpha()])
print('Number of words (not including punctuation): ', number_words)

# avg number of words/per sentence
avg_words_per_sentence = number_words / len(brown.sents())
print('Avg number of words per sentence: ', avg_words_per_sentence)

#avg word length
avg_word_length = sum(len(word) for word in brown.words() if word.isalpha()) / number_words
print('Avg word length: ', avg_word_length)

#POS tagging, but exclude punctuations from NLTK library
tag_words = pos_tag([word for word in brown.words() if word.isalpha()])

#count the frequency of the words so you can then list them properly
pos_counts = Counter(tag for word, tag in tag_words)

#then finally list top ten frequent POS tags
most_common_pos = pos_counts.most_common(10)

print('most ocmmon ten POS tags (excluding punctuation): ', most_common_pos)

##############
####### Plots of frequencies

def plot_freq(sorted_words, title):
    frequencies = [freq for word, freq in sorted_words]
    positions = range(1, len(sorted_words) + 1)

    #linear axes
    plt.figure(figsize=(10, 6))
    plt.plot(positions, frequencies, label='Frequency Curves (linear)', color='#20B2AA')
    plt.xlabel('position in frequency list')
    plt.ylabel('Frequency')
    plt.title(title + ' (linear)')
    plt.legend()
    plt.grid(True)
    plt.show()

    #log scale axes
    plt.figure(figsize=(10, 6))
    plt.loglog(positions, frequencies, label='frequency Curve (log scale)', color='#D8BFD8')
    plt.xlabel('position in frequency list (log)')
    plt.ylabel('Frequency (log scale)')
    plt.title(title + ' (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()


#brown corpus
plot_freq(sorted_words, 'Frequency curve of Brown Corpus')

#frequencies of humor and mystery
plot_freq(sorted_both_words, 'Frequency Curves of Humor and Mystery')