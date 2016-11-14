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
    document_file = []
    with open(path, 'r') as f:
        # lignes = f.readlines()
        # lignes = [l[:-1] for l in lignes]
        lignes = [line.decode('utf-8').strip() for line in f.readlines()]
        for i in range(0, len(lignes), 7):
            document_file = [defaultdict(dict) for _ in range(0, len(lignes), 7)]
        # Todo: Change the structure overhere because it doesnt work prettywell. Maybe i should create a big json file
        # Todo: and build the index byloping over the json file
        j = 0
        for i in range(len(document_file)):
            document_file[i]['id'] = lignes[j]
            document_file[i]['key'] = lignes[j + 2]
            document_file[i]['title'] = lignes[j + 4]
            document_file[i]['content'] = lignes[j + 6]
            j += 7
    return document_file
    # print document_db.keys()
    # with open("document_db.json", "w") as fich:
    #     json.dump(document_db, fich)


def stemming_processing(data_base):
    """ Decomposition and stemming of the word into the test and rebuild a new dictionnary with the stemming word"""
    stemmer = nltk.stem.PorterStemmer()
    for k in data_base.keys():
        corpus = data_base[k]['content']
        # Decomposition
        tokens = nltk.word_tokenize(corpus)
        # Stemming
        stems = [stemmer.stem(mot) for mot in tokens]
        data_base[k]['content'] = stems


def load_query(path):
    """ Load the query file and build a list of the query"""
    with open(path, 'r') as f:
        lignes = [line.decode('utf-8').strip().split('\t') for line in f.readlines()]
    return lignes  # list of list


def information_retrival():
    """ Use elastic net to do the operation! J'ai trouvé un note book sur internet, le suivre à la lettre car il montre
    que j'ai pas beoin de creer plusieurs index par exemple
    https://github.com/ernestorx/es-swapi-test/blob/master/ES%20notebook.ipynb . Juste incrémenter le id number"""
    from elasticsearch import Elasticsearch
    es = Elasticsearch()
    doc_db_list = load_dataset(path="/home/maoss2/Documents/Doctorat/Automne2016/NLP/TP2/file_collection.txt")
    res = None
    for i, el in enumerate(doc_db_list[:10]):
        res = es.index(index="nlp_tp2", doc_type='document', id=i, body=el, request_timeout=100000)
    # print(res['created'])
    # print res

    # get the index
    # res = es.get(index="nlp_tp2", doc_type='document', id=6)
    # print(res['_source'])

    # load the query 1st
    list_of_queries = load_query(path="/home/maoss2/Documents/Doctorat/Automne2016/NLP/TP2/list_queries.txt")
    es.indices.refresh(index="nlp_tp2")
    search_results = []

    # make the research by looping over every queries
    for query in list_of_queries:
        search_results.append(es.search(index="nlp_tp2", body={"query": {"match": {'content': query[1]}}}))

    for i, r in enumerate(search_results):
        print list_of_queries[i][1]
        print_results(r)
    return list_of_queries

def print_results(res):
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
        print("%(key)s: %(content)s" % hit["_source"])
        # print("%(key)s %(title)s: %(content)s" % hit["_source"])

def mean_average_precision(y_true, y_pred):
    """Calculate the map. Pseudo code: calcul le average precision with sklearn package then
     build the mean of the list"""
    # Todo: Make a general function for this

def pseudo_relevance_feedback():
    """ Select some k 1st element then build the rochio algorithm"""
    # Todo: Make a general function for this

if __name__ == '__main__':
    information_retrival()
    # document_db = load_dataset(path="/home/maoss2/Documents/Doctorat/Automne2016/NLP/TP2/file_collection.txt")
    # stemming_processing(data_base=document_db)
    # print document_db.values()
