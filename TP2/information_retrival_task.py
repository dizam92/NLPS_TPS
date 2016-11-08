# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import nltk
from joblib import Parallel, delayed


# nltk.download()

def load_dataset(path):
    """ Load the the file collection document """
    document_db = defaultdict(dict)
    with open(path, 'r') as f:
        # lignes = f.readlines()
        # lignes = [l[:-1] for l in lignes]
        lignes = [line.decode('utf-8').strip() for line in f.readlines()]
        for i in range(0, len(lignes), 7):
            document_db['doc_%d' % i]['id'] = lignes[i]
            document_db['doc_%d' % i]['key'] = lignes[i + 2]
            document_db['doc_%d' % i]['title'] = lignes[i + 4]
            document_db['doc_%d' % i]['content'] = lignes[i + 6]
    return document_db
    # print document_db.keys()
    # with open("document_db.json", "w") as fich:
    #     json.dump(document_db, fich)


def stemming_processing(k, data_base):
    """ Decomposition and stemming of the word into the test and rebuild a new dictionnary with the stemming word"""
    stemmer = nltk.stem.PorterStemmer()
    corpus = data_base[k]['content']
    # Decomposition
    tokens = nltk.word_tokenize(corpus)
    # Stemming
    stems = [stemmer.stem(mot) for mot in tokens]
    data_base[k]['content'] = stems

def information_retrival():
    
if __name__ == '__main__':
    document_db = load_dataset(path="/home/maoss2/Documents/Doctorat/Automne2016/NLP/TP2/file_collection.txt")
    Parallel(n_jobs=8)(delayed(stemming_processing)(k, data_base=document_db) for k in document_db.keys())
    print document_db.values()
