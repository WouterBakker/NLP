import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter, OrderedDict



### Compute a list of unique words sorted by descending frequency for entire corpus
# Counts of each words
all_words = brown.words()
all_words_decapitalized = [x.lower() for x in all_words]

wordcounts = Counter(all_words_decapitalized)
# most frequent words
sorted_wordcounts = dict(wordcounts.most_common())

# brown.categories() #show categories in brown corpus

### Compute a list of unique words sorted by descending frequency for two categories
# subset two categories
genre_adventure_with_capitals = brown.words(categories='adventure')   
genre_adventure = [x.lower() for x in genre_adventure_with_capitals] #remove capital letters for proper counting
genre_news_with_capitals = brown.words(categories='news')   
genre_news = [x.lower() for x in genre_news_with_capitals] #remove capital letters for proper counting

# most frequent words
sorted_wordcounts_adventure = dict(Counter(genre_adventure).most_common())
sorted_wordcounts_news = dict(Counter(genre_news).most_common())


# The number of tokens equals the total number of word instantiations
n_tokens = len(all_words_decapitalized)
print(f"Number of tokens: {n_tokens}")

# The number types, aka the number of unique words
n_types = len(wordcounts)
print(f"Number of types: {n_types}")

# The number of words 
print(f"Number of words: {len(all_words)}")


# Average nr of words per sentence
avg_len_sents = len(brown.words()) / len(brown.sents())
print(f"average number of words per sentence: {avg_len_sents}")


# Average word length
avg_len_word = sum([len(x) for x in brown.words()])/len(brown.words())
print(f"Average word length: {avg_len_word}")


# Run POS tagger
pos_tuples = nltk.pos_tag(brown.words())

dd = defaultdict(int)

for x,y in pos_tuples:
    dd[y] += 1

print(f"10 most common POS tags: {Counter(dd).most_common()[:10]}")






## Plotting the word frequencies

# Frequencies of all words
x_all = list(range(1, len(sorted_wordcounts)+1))
y_all = list(sorted_wordcounts.values())

# Frequency of words in adventure genre
x_adventure = list(range(1, len(sorted_wordcounts_adventure)+1))
y_adventure = list(sorted_wordcounts_adventure.values())

# Frequency of words in news genre
x_news= list(range(1, len(sorted_wordcounts_news)+1))
y_news = list(sorted_wordcounts_news.values())


import matplotlib.pyplot as plt

plt.suptitle('Word Frequencies for:', fontsize=16)

# First row: Original data
plt.subplot(2, 3, 1)
plt.plot(x_all, y_all, marker='o', linestyle='-', color='b')  # Plot the frequency curve
plt.title('All Words/Genres in Corpus')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x_adventure, y_adventure, marker='o', linestyle='-', color='b')  # Plot the frequency curve
plt.title('Adventure genre')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x_news, y_news, marker='o', linestyle='-', color='b')  # Plot the frequency curve
plt.title('News genre')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

# Second row: Log-log transformed data
plt.subplot(2, 3, 4)
plt.loglog(x_all, y_all, marker='o', linestyle='-', color='b')  # Plot the log-log frequency curve
plt.title('Log-log: All Genres in Corpus')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.loglog(x_adventure, y_adventure, marker='o', linestyle='-', color='b')  # Plot the log-log frequency curve
plt.title('Log-log: Adventure genre')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.loglog(x_news, y_news, marker='o', linestyle='-', color='b')  # Plot the log-log frequency curve
plt.title('Log-log: News genre')
plt.xlabel('Word position')
plt.ylabel('Frequency')
plt.grid(True)

# Adjust layout
# plt.tight_layout()

# Show the plots
plt.show()