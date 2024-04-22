from nltk.corpus import brown
from collections import defaultdict, Counter, OrderedDict

# Compute a list of unique words sorted by descending frequency for:
## (i) the whole corpus 
wordcounts = Counter(brown.words())
sorted_wordcounts = dict(wordcounts.most_common())


## (ii) two different genres of your choice.

# brown.categories()

genre_adventure = brown.words(categories='adventure')   
genre_news = brown.words(categories='news')   

sorted_wordcounts_adventure = dict(Counter(genre_adventure).most_common())
sorted_wordcounts_newss = dict(Counter(genre_news).most_common())
