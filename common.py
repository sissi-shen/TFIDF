from tfidf import *
from collections import Counter
import sys

"""
Print the most common 10 words from a documents and the word count.

1. Use gettext to get the text of the xml file.
2. Tokenize the text with tokenize.
3. Compute word counts with Counter.
4. Print most common words with counts.

$ python common.py ~/data/reuters-vol1-disk1-subset/33313newsML.xml
power 14
transmission 14
new 12
say 12
generator 12
electricity 11
cost 10
zealand 9
signal 8
charge 7
"""

path = sys.argv[1]

text = gettext(path)
nlp = spacy.load("en_core_web_sm")
tok_doc = tokenize(text, nlp)
word_count = Counter(tok_doc)
top_10 = word_count.most_common(10)
for word, count in top_10:
    print(f"{word}: {count}")