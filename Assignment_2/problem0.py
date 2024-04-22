from nltk.corpus import brown
from collections import defaultdict, Counter, OrderedDict


wordcounts = Counter(brown.words())
sorted_wordcounts = dict(wordcounts.most_common())
