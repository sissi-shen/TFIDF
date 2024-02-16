from tfidf import *
import os
import pickle

"""
1. Get a list of all xml_files in the corpus (~/data/reuters-vol1-disk1-subset).
2. Get a list of texts for all files xml_files.
3. Get a list of tokenized text (list of list of tokens).
4. Save the tokenized corpus in ~/data/tok_corpus.pickle.
"""
directory_path = sys.argv[1]
xml_dir = os.path.expanduser(directory_path) 
xml_files = [f for f in os.listdir(xml_dir)]
testfile_paths = [os.path.join(xml_dir, file) for file in xml_files]

text_corpus = [gettext(f) for f in testfile_paths]
nlp = spacy.load("en_core_web_sm")
tok_corpus = [tokenize(text, nlp) for text in text_corpus]

pickle_file = os.path.expanduser("~/data/tok_corpus.pickle")
with open(pickle_file, 'wb') as file:
    pickle.dump(tok_corpus, file)
