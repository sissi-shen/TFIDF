import sys

import spacy
from lxml import etree
from collections import Counter
import string
import zipfile
import os
import numpy as np
import re

"""
pytest -v

test_tfidf.py::test_gettext PASSED                                                              [ 14%]
test_tfidf.py::test_tokenize PASSED                                                             [ 28%]
test_tfidf.py::test_tokenize_2 PASSED                                                           [ 42%]
test_tfidf.py::test_doc_freq PASSED                                                             [ 57%]
test_tfidf.py::test_compute_tfidf_i PASSED                                                      [ 71%]
test_tfidf.py::test_compute_tfidf PASSED                                                        [ 85%]
test_tfidf.py::test_summarize PASSED                                                            [100%]
"""

def gettext(xmlfile) -> str:
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    tree = etree.parse(xmlfile)
    root = tree.getroot()

    title = root.xpath('.//title/text()')
    text = root.xpath('.//text/p/text()')
    
    title = ''.join(title)
    text = ' '.join(text)
    text_str = title + ' ' + text

    return text_str


def tokenize(text, nlp) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. 
      1. Normalize to lowercase. Strip punctuation, numbers, and `\r`, `\n`, `\t`. 
      2. Replace multiple spaces for a single space.
      3. Tokenize with spacy.
      4. Remove stopwords with spacy.
      5. Remove tokens with len <= 2.
      6. Apply lemmatization to words using spacy.
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    token_list = []
    pos = ['PUNCT', 'SPACE']
    for token in doc:
        if len(str(token).strip()) > 2 and token.is_stop != True:
            token_list.append(token.lemma_)
        
    return token_list


def doc_freq(tok_corpus):
    """
    Returns a dictionary of the number of docs in which a word occurs.
    Input:
       tok_corpus: list of list of words
    Output:
       df: dictionary df[w] = # of docs containing w 
    """
    df = {}
    for doc in tok_corpus:
        unique_doc = set(doc)
        for word in unique_doc:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1

    return df


def compute_tfidf_i(tok_doc: list, doc_freq: dict, N: int) -> dict:
    """ Returns a dictionary of tfidf for one document
        tf[w, doc] = counts[w, doc]/ len(doc)
        idf[w] = np.log(N/(doc_freq[w] + 1))
        tfidf[w, doc] = tf[w, doc]*idf[w]
    """
    tfidf = {}

    word_count = Counter(tok_doc)
    total_words = len(tok_doc)
    tf = {word: count / total_words for word, count in word_count.items()}

    for word in tf:
        if word in doc_freq:
            idf = np.log(N / (doc_freq[word] + 1))
            tfidf[word] = tf[word]*idf
        else:
            tfidf[word] = 0.0

    return tfidf


def compute_tfidf(tok_corpus:list, doc_freq: dict) -> dict:
    """Computes tfidf for a corpus of tokenized text.

    Input:
       tok_corpus: list of tokenized text
       doc_freq: dictionary of word to set of doc indeces
    Output:
       tfidf: list of dict 
               tfidf[i] is the dictionary of tfidf of word in doc i.
    """
    tfidf = []
    N = len(tok_corpus)

    for tok_doc in tok_corpus:
        tfidf_dict = compute_tfidf_i(tok_doc, doc_freq, N)
        tfidf.append(tfidf_dict)

    return tfidf


def summarize(xmlfile, doc_freq, N,  n:int) -> list:
    """
    Given xml file, n and the tfidf dictionary 
    return up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.01. Sort the (word,score) pairs by TFIDF score in reverse order.
    if words have the same score, they should be sorted in alphabet order.
    """
    text = gettext(xmlfile)
    nlp = spacy.load("en_core_web_sm")
    tok_corpus = tokenize(text, nlp)

    tfidf_dict = compute_tfidf_i(tok_corpus, doc_freq, N)

    top_n = sorted(
        [(word, score) for word, score in tfidf_dict.items() if score >= 0.01],
        key=lambda x: (-x[1], x[0])
        )[:n]

    return top_n


