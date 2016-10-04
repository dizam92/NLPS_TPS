# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import pandas as pd
import csv
import re
import ast
from collections import defaultdict
from time import time

start_phrase = "<s>"
end_phrase = "</s>"
def count_token():
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        number_of_token = np.sum([len(d) for d in data])
    return number_of_token

def count_1_gram():
    """ New idea: Build a datastructure to keep the trigram, bigram, and unigram together """
    unigram_count_dictionnary = defaultdict(int)
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        for d in data:
            d.insert(0, start_phrase)
            d.append(end_phrase)
            for i in xrange(len(d)):
                uni_key = d[i]
                if uni_key not in unigram_count_dictionnary.keys():
                    unigram_count_dictionnary[uni_key] = 1
                else:
                    unigram_count_dictionnary[uni_key] += 1

    with open('unigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, unigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(unigram_count_dictionnary)

def count_2_gram():
    bigram_count_dictionnary = defaultdict(int)
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        for d in data:
            d.insert(0, start_phrase)
            d.insert(0, start_phrase)
            d.append(end_phrase)
            d.append(end_phrase)
            for i in xrange(len(d) - 1):
                bin_key = d[i] + " " + d[i + 1]
                if bin_key not in bigram_count_dictionnary.keys():
                    bigram_count_dictionnary[bin_key] = 1
                else:
                    bigram_count_dictionnary[bin_key] += 1
    with open('bigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, bigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(bigram_count_dictionnary)

def count_3_gram():
    """ New idea: Build a datastructure to keep the trigram, bigram, and unigram together """
    trigram_count_dictionnary = defaultdict(int)
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        for d in data:
            d.insert(0, start_phrase)
            d.insert(0, start_phrase)
            d.append(end_phrase)
            d.append(end_phrase)
            for i in xrange(len(d) - 2):
                tri_key = d[i] + " " + d[i + 1] + " " + d[i + 2]
                if tri_key not in trigram_count_dictionnary.keys():
                    trigram_count_dictionnary[tri_key] = 1
                else:
                    trigram_count_dictionnary[tri_key] += 1
    with open('trigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, trigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(trigram_count_dictionnary)

def linked_dictionnary_builder(unigram_file, bigram_file, trigram_file):
    """ New idea: Build a datastructure to keep the trigram, bigram, and unigram together """
    final_dictionnary = defaultdict(list)
    data_bi = pd.read_csv(bigram_file)
    data_uni = pd.read_csv(unigram_file)
    data_tri = pd.read_csv(trigram_file)

    for uni_key in data_uni.columns.values:
        list_bin = [bin_key for bin_key in np.sort(data_bi.columns.values) if uni_key == bin_key.split(" ")[0] or uni_key == bin_key.split(" ")[1]]
        list_tri = [tri_key for tri_key in np.sort(data_tri.columns.values) if uni_key == tri_key.split(" ")[0] or uni_key == tri_key.split(" ")[1] or uni_key == tri_key.split(" ")[2]]
        final_dictionnary[uni_key] = [list_bin, list_tri]

    # write dictionnay to file
    with open('finaldictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, final_dictionnary.keys())
        w.writeheader()
        w.writerow(final_dictionnary)

def count_n_gram():
    print "Counting 1 gram......"
    count_1_gram()
    print "Counting 2 gram......"
    count_2_gram()
    print "Counting 3 gram......"
    count_3_gram()
    print "Linking the n gram keys......"
    linked_dictionnary_builder(unigram_file="unigram_count_dictionnary.csv",
                               bigram_file="bigram_count_dictionnary.csv",
                               trigram_file="trigram_count_dictionnary.csv")

def unigram_model(list_of_words, unigram_count, N=count_token()):
    """ Compute the MLE for unigram model
    Params: list_of_words: list of the unique word to be learned
            unigram_count: dictionnary of the count of the words in the corpus"""
    d = pd.read_csv(unigram_count)
    # proba_matrix = defaultdict(float)
    proba_dict = {list_of_words[i]: (d[el].values[0] / float(N)) if el in d.columns.values else 0.0 for i, el in enumerate(list_of_words) }
    return proba_dict

def laplace_delta_unigram_model(list_of_words, unigram_count, N=count_token(), delta=None):
    """ Apply the lissage delta """
    assert 0 <= delta <= 1, "Delta must be between 0 and 1"
    d = pd.read_csv(unigram_count)
    proba_dict = {list_of_words[i]: ((d[el].values[0] + delta) / float(N + d.columns.values.size))
                        if el in d.columns.values else (delta / float(N + d.columns.values.size)) for i, el in enumerate(list_of_words)}
    return proba_dict

def interpolation_unigram_model(list_of_words, unigram_count, N=count_token(), lambda3=None):
    """ Apply the interpolation lissage"""
    assert 0 < lambda3 <= 1, "wrong value"
    d = pd.read_csv(unigram_count)
    proba_dict = {list_of_words[i]: lambda3 * (d[el].values[0] / float(N)) if el in d.columns.values else 0.0 for i, el in
                  enumerate(list_of_words)}
    return proba_dict

def bigram_model(list_of_words, unigram_count, bigram_count):
    """ Compute the MLE for bigram model
        Params: list_of_words: list of the unique word to be learned
                unigram_count: dictionnary of the count of the words in the corpus
                bigram_count: dictionnary of the count of the bi-words in the corpus"""
    if start_phrase not in list_of_words:
        list_of_words.insert(0, start_phrase)
    if end_phrase not in list_of_words:
        list_of_words.append(end_phrase)
    uni_count = pd.read_csv(unigram_count)
    bigram_count = pd.read_csv(bigram_count)
    # proba_matrix = defaultdict(float)
    proba_dict = {list_of_words[i] + " " + list_of_words[i+1]: (bigram_count[list_of_words[i] + " " + list_of_words[i+1]].values[0] / float(uni_count[list_of_words[i]].values[0]))
                    if list_of_words[i] + " " + list_of_words[i+1] in bigram_count.columns.values else 0.0 for i in xrange(len(list_of_words) - 1)}
    return proba_dict
    # for i in xrange(len(list_of_words) - 1):
    #     bi_words = list_of_words[i] + " " + list_of_words[i+1]
    #     if bi_words in bigram_count.columns.values:
    #         proba_matrix = {bi_words: (bigram_count[bi_words] / float(list_of_words[i]))}
    #     else:
    #         proba_matrix = {bi_words: 0.0}

def laplace_delta_bigram_model(list_of_words, unigram_count, bigram_count, delta=None):
    """ Apply the lissage delta """
    assert 0 <= delta <= 1, "Delta must be between 0 and 1"
    if start_phrase not in list_of_words:
        list_of_words.insert(0, start_phrase)
    if end_phrase not in list_of_words:
        list_of_words.append(end_phrase)
    uni_count = pd.read_csv(unigram_count)
    bigram_count = pd.read_csv(bigram_count)
    proba_dict = {list_of_words[i] + " " + list_of_words[i + 1]:
                        ((bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] + delta) /
                         float(uni_count[list_of_words[i]].values[0] + uni_count.columns.values.size))
                    if list_of_words[i] + " " + list_of_words[i + 1] in bigram_count.columns.values
                    else (delta / float(uni_count[list_of_words[i]].values[0] + uni_count.columns.values.size))
                    for i in xrange(len(list_of_words) - 1)}
    return proba_dict

def interpolation_bigram_model(list_of_words, unigram_count, bigram_count, N=count_token(), lambda2=None, lambda3=None):
    """ Apply the interpolation lissage"""
    assert 0 < lambda3 <= 1, "wrong value"
    assert 0 < lambda2 <= 1, "wrong value"
    assert 0 < lambda2 + lambda3 <= 1, "wrong value"
    if start_phrase not in list_of_words:
        list_of_words.insert(0, start_phrase)
    if end_phrase not in list_of_words:
        list_of_words.append(end_phrase)
    uni_count = pd.read_csv(unigram_count)
    bigram_count = pd.read_csv(bigram_count)
    proba_dict = {list_of_words[i] + " " + list_of_words[i + 1]:
            lambda2 * (bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] / float(uni_count[list_of_words[i]].values[0])) +
            lambda3 * (uni_count[list_of_words[i]].values[0] / float(N))
            if list_of_words[i] + " " + list_of_words[i + 1] in bigram_count.columns.values
            else lambda3 * (uni_count[list_of_words[i]].values[0] / float(N)) for i in xrange(len(list_of_words) - 1)}
    return proba_dict

def trigram_model(list_of_words, bigram_count, trigram_count):
    """ Compute the MLE for bigram model
        Params: list_of_words: list of the unique word to be learned
                unigram_count: dictionnary of the count of the words in the corpus
                bigram_count: dictionnary of the count of the bi-words in the corpus
                trigram_count: dictionnary of the count of the tri-words in the corpus"""
    c_start = list_of_words.count(start_phrase)
    c_end = list_of_words.count(end_phrase)
    if c_start == 0:
        list_of_words.insert(0, start_phrase)
        list_of_words.insert(0, start_phrase)
    if c_start == 1:
        list_of_words.insert(0, start_phrase)
    if c_end == 0:
        list_of_words.append(end_phrase)
        list_of_words.append(end_phrase)
    if c_end == 1:
        list_of_words.append(end_phrase)
    bigram_count = pd.read_csv(bigram_count)
    trigram_count = pd.read_csv(trigram_count)
    proba_dict = {list_of_words[i] + " " + list_of_words[i+1] + " " + list_of_words[i+2]:
        ((trigram_count[list_of_words[i] + " " + list_of_words[i+1] + " " + list_of_words[i+2]].values[0]) /
                         float(bigram_count[list_of_words[i] + " " + list_of_words[i+1]].values[0]))
                     if list_of_words[i] + " " + list_of_words[i+1] + " " + list_of_words[i+2] in trigram_count.columns.values else 0.0 for i in xrange(len(list_of_words) - 2)}
    return proba_dict

def laplace_delta_trigram_model(list_of_words, unigram_count, bigram_count, trigram_count, delta=None):
    """ Apply the lissage delta """
    assert 0 <= delta <= 1, "Delta must be between 0 and 1"
    c_start = list_of_words.count(start_phrase)
    c_end = list_of_words.count(end_phrase)
    if c_start == 0:
        list_of_words.insert(0, start_phrase)
        list_of_words.insert(0, start_phrase)
    if c_start == 1:
        list_of_words.insert(0, start_phrase)
    if c_end == 0:
        list_of_words.append(end_phrase)
        list_of_words.append(end_phrase)
    if c_end == 1:
        list_of_words.append(end_phrase)
    uni_count = pd.read_csv(unigram_count)
    bigram_count = pd.read_csv(bigram_count)
    trigram_count = pd.read_csv(trigram_count)
    proba_dict = {list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]:
            ((trigram_count[list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]].values[0] + delta) /
            float(bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] + uni_count.columns.values.size))
            if list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2] in trigram_count.columns.values
            else (delta / float(bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] + uni_count.columns.values.size))
            for i in xrange(len(list_of_words) - 2)}
    return proba_dict

def interpolation_trigram_model(list_of_words, unigram_count, bigram_count, trigram_count, N=count_token(), lambda1=None, lambda2=None, lambda3=None):
    """ Apply the interpolation lissage"""
    assert 0 < lambda3 <= 1, "wrong value"
    assert 0 < lambda2 <= 1, "wrong value"
    assert 0 < lambda1 <= 1, "wrong value"
    assert 0 < lambda1 + lambda2 + lambda3 <= 1, "wrong value"
    c_start = list_of_words.count(start_phrase)
    c_end = list_of_words.count(end_phrase)
    if c_start == 0:
        list_of_words.insert(0, start_phrase)
        list_of_words.insert(0, start_phrase)
    if c_start == 1:
        list_of_words.insert(0, start_phrase)
    if c_end == 0:
        list_of_words.append(end_phrase)
        list_of_words.append(end_phrase)
    if c_end == 1:
        list_of_words.append(end_phrase)
    uni_count = pd.read_csv(unigram_count)
    bigram_count = pd.read_csv(bigram_count)
    trigram_count = pd.read_csv(trigram_count)
    proba_dict = {list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]:
                      lambda1 * ((trigram_count[list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]].values[0]) / float(bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0])) +
                      lambda2 * (bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] / float(uni_count[list_of_words[i]].values[0])) +
                      lambda3 * (uni_count[list_of_words[i]].values[0] / float(N))
                      if list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2] in trigram_count.columns.values
                      else lambda2 * (bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] / float(uni_count[list_of_words[i]].values[0])) +
                           lambda3 * (uni_count[list_of_words[i]].values[0] / float(N))
                           if list_of_words[i] + " " + list_of_words[i + 1] in bigram_count.columns.values
                           else lambda3 * (uni_count[list_of_words[i]].values[0] / float(N)) for i in xrange(len(list_of_words) - 2)}
    return proba_dict

def probability_builder(n, number_of_token):
    """ Build the matrix of the probability for the 1, 2, 3 gram"""
    if n == 1:
        d = pd.read_csv("unigram_count_dictionnary.csv")
        d.columns.values.sort()
        unigram_probability = d.values / float(number_of_token)
        temp_list = [(index,cle) for index, cle in enumerate(np.sort(d.columns.values))]
        return unigram_probability, temp_list

    # Explications: Construire une matrice carré de words² elements. Pour chaque mot (Vu que c'est trié c'est dans le meme ordre)
    # regarder si le mot se retrouve au niveau w_(i-1) du bigram soit la position 2 dans le bigram,
    # si oui, on calcule le c(w_i, w_(i-1))/c(w_i) et mettre dans la matrice à la position (w_(i-1), w_i)
    if n == 2:
        linked_words = pd.read_csv("finaldictionnary.csv")
        bigram_proba_arrays = np.zeros((linked_words.columns.size, linked_words.columns.size), dtype=float)
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        for index, cle in enumerate(np.sort(linked_words.columns.values)):
            bigram_list = ast.literal_eval(linked_words[cle][0])[0]  # get the list of the bigram only
            for el in bigram_list:
                words = el.split(" ")
                if words[1] == cle: # look if 2nd word is key
                    bigram_proba_arrays[index, np.where(linked_words.columns.values == words[0])[0][0]] = data_bi[el] / float(data_uni[words[0]])
        temp_list = [(index, cle) for index, cle in enumerate(np.sort(data_uni.columns.values))]
        return bigram_proba_arrays, temp_list

    if n == 3:
        linked_words = pd.read_csv("finaldictionnary.csv")
        data_tri = pd.read_csv("trigram_count_dictionnary.csv")
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        # data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        trigram_proba_arrays = np.zeros((linked_words.columns.size, data_bi.columns.size), dtype=float)
        for index, cle in enumerate(np.sort(linked_words.columns.values)):
            trigram_list = ast.literal_eval(linked_words[cle][0])[1]  # get the list of the bigram only
            for el in trigram_list:
                words = el.split(" ")
                if words[2] == cle:
                    trigram_proba_arrays[index, np.where(np.sort(data_bi.columns.values) == words[words[0] + " " + words[1]])[0][0]] = \
                        data_tri[el] / float(data_bi[words[0] + " " + words[1]])

        temp_list_ligne = [(index, cle) for index, cle in enumerate(np.sort(linked_words.columns.values))]
        temp_list_col = [(index, cle) for index, cle in enumerate(np.sort(data_bi.columns.values))]
        return trigram_proba_arrays, [temp_list_ligne, temp_list_col]

def lissage_add_delta(n, number_of_token, delta):
    """ Apply the add_delta lissage to the matrix"""
    assert 0 <= delta <= 1, "Delta must be between 0 and 1"
    if n == 1:
        d = pd.read_csv("unigram_count_dictionnary.csv")
        d.columns.values.sort()
        unigram_probability = (d.values + delta) / float((number_of_token + d.columns.values.size))
        temp_list = [(index, cle) for index, cle in enumerate(np.sort(d.columns.values))]
        return unigram_probability, temp_list

    if n == 2:
        linked_words = pd.read_csv("finaldictionnary.csv")
        bigram_proba_arrays = np.zeros((linked_words.columns.size, linked_words.columns.size), dtype=float)
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        for index1, cle1 in enumerate(np.sort(linked_words.columns.values)):
            for index2, cle2 in enumerate(np.sort(linked_words.columns.values)):
                temp_word = cle1 + " " + cle2
                if temp_word in data_bi.columns.values:
                    bigram_proba_arrays[index1, index2] = (data_bi[temp_word] + delta) / float((data_uni[cle1] + data_uni.columns.values.size))
                else:
                    bigram_proba_arrays[index1, index2] = 1 / float((data_uni[cle1] + data_uni.columns.values.size))
        temp_list = [(index, cle) for index, cle in enumerate(np.sort(data_uni.columns.values))]
        return bigram_proba_arrays, temp_list

    if n == 3:
        linked_words = pd.read_csv("finaldictionnary.csv")
        data_tri = pd.read_csv("trigram_count_dictionnary.csv")
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        # data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        trigram_proba_arrays = np.zeros((linked_words.columns.size, data_bi.columns.size), dtype=float)
        for index1, cle1 in enumerate(np.sort(linked_words.columns.values)):
            for index2, cle2 in enumerate(np.sort(data_bi.columns.values)):
                temp_word = cle1 + " " + cle2
                if temp_word in data_tri.columns.values:
                    trigram_proba_arrays[index1, index2] = \
                        (data_tri[temp_word] + delta) / float((data_bi[temp_word] + linked_words.columns.values.size))
                else:
                    trigram_proba_arrays[index1, index2] = 1 / float((data_bi[temp_word] + linked_words.columns.values.size))
        temp_list_ligne = [(index, cle) for index, cle in enumerate(np.sort(linked_words.columns.values))]
        temp_list_col = [(index, cle) for index, cle in enumerate(np.sort(data_bi.columns.values))]
        return trigram_proba_arrays, [temp_list_ligne, temp_list_col]


def lissage_interpolation(n, number_of_token, lambda1=0.001, lambda2=0.001, lambda3=0.001):
    """ Apply the interpolation lissage"""
    if n == 1:
        assert 0 < lambda3 <= 1, "wrong value"
        d = pd.read_csv("unigram_count_dictionnary.csv")
        d.columns.values.sort()
        unigram_probability = lambda3 * (d.values / float(number_of_token))
        temp_list = [(index, cle) for index, cle in enumerate(np.sort(d.columns.values))]
        return unigram_probability, temp_list

    if n == 2:
        assert 0 < lambda3 <= 1, "wrong value"
        assert 0 < lambda2 <= 1, "wrong value"
        assert 0 < lambda2 + lambda3 <= 1, "wrong value"
        data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        linked_words = pd.read_csv("finaldictionnary.csv")
        bigram_proba_arrays = np.zeros((linked_words.columns.size, linked_words.columns.size), dtype=float)
        for index1, cle1 in enumerate(np.sort(linked_words.columns.values)):
            for index2, cle2 in enumerate(np.sort(linked_words.columns.values)):
                temp_word = cle1 + " " + cle2
                if temp_word in data_bi.columns.values:
                    bigram_proba_arrays[index1, index2] = lambda2 * (data_bi[temp_word] / float(data_uni[cle1])) + \
                                                          lambda3 * (data_uni[cle1] / float(number_of_token))
                else:
                    bigram_proba_arrays[index1, index2] = lambda3 * (data_uni[cle1] / float(number_of_token))
        temp_list = [(index, cle) for index, cle in enumerate(np.sort(data_uni.columns.values))]
        return bigram_proba_arrays, temp_list

    if n == 3:
        assert 0 < lambda3 <= 1, "wrong value"
        assert 0 < lambda2 <= 1, "wrong value"
        assert 0 < lambda1 <= 1, "wrong value"
        assert 0 < lambda1+lambda2+lambda3 <= 1, "wrong value"
        data_tri = pd.read_csv("trigram_count_dictionnary.csv")
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        trigram_proba_arrays = np.zeros((data_uni.columns.size, data_bi.columns.size), dtype=float)
        for index1, cle1 in enumerate(np.sort(data_uni.columns.values)):
            for index2, cle2 in enumerate(np.sort(data_bi.columns.values)):
                temp_word = cle1 + " " + cle2
                if temp_word in data_tri.columns.values:
                    trigram_proba_arrays[index1, index2] = lambda1 * data_tri[temp_word] / float(data_bi[cle2]) + \
                                                           lambda2 * (data_bi[cle2] / float(data_uni[cle1])) + \
                                                           lambda3 * (data_uni[cle1] / float(number_of_token))
                elif cle2 in data_bi.columns.values:
                    trigram_proba_arrays[index1, index2] = lambda2 * (data_bi[cle2] / float(data_uni[cle1])) + \
                                                            lambda3 * (data_uni[cle1] / float(number_of_token))
                else:
                    trigram_proba_arrays[index1, index2] = lambda3 * (data_uni[cle1] / float(number_of_token))
        temp_list_ligne = [(index, cle) for index, cle in enumerate(np.sort(linked_words.columns.values))]
        temp_list_col = [(index, cle) for index, cle in enumerate(np.sort(data_bi.columns.values))]
        return trigram_proba_arrays, [temp_list_ligne, temp_list_col]

def perplexity_calcul(n, testset, model, list_unigram=None, list_bigram=None, list_trigram=None):
    if n == 1:
        index, cle = zip(*list_unigram)
        index = np.asarray(index)
        cle = np.asarray(cle)
        # testset = testset.split(" ")
        log_proba = 0
        N = 0
        for word in testset:
            N += 1
            pos = np.where(cle == word)[0][0]
            log_proba += np.log2(model[:, pos])
        perplexity = np.exp(- log_proba / float(N))
        return perplexity

    if n == 2:
        index, cle = zip(*list_bigram)
        index = np.asarray(index)
        cle = np.asarray(cle)
        # testset = testset.split(" ")
        log_proba = 0
        N = 0
        for i in xrange(len(testset)-1):
            N += 1
            lign = np.where(cle == testset[i])[0][0]
            col = np.where(cle == testset[i+1])[0][0]
            log_proba += np.log2(model[lign][col])
        perplexity = np.exp(- log_proba / float(N))
        return perplexity

    if n == 3:
        index_ligne, cle_ligne = zip(*list_trigram[0])
        index_col, cle_col = zip(*list_trigram[1])
        index_ligne = np.asarray(index_ligne)
        cle_ligne = np.asarray(cle_ligne)
        index_col = np.asarray(index_col)
        cle_col = np.asarray(cle_col)
        # testset = testset.split(" ")
        log_proba = 0
        N = 0
        for i in xrange(len(testset) - 2):
            N += 1
            biwords = testset[i + 1] + testset[i + 2]
            lign = np.where(cle_ligne == testset[i])[0][0]
            col = np.where(cle_col == biwords)[0][0]
            log_proba += np.log2(model[lign][col])
        perplexity = np.exp(- log_proba / float(N))
        return perplexity

def unigram_experience(fichier_test):
    proba_matrix, indx_and_names = probability_builder(2, number_of_token=count_token())
    proba_lissage_matrix, indx_and_names_lissages = lissage_add_delta(2, number_of_token=count_token(), delta=0.05)
    proba_interpolation_matrix, indx_and_names_interpolation = lissage_interpolation(2, number_of_token=count_token(), lambda1=0.05)
    exit()
    with open(fichier_test, 'r') as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        data = data[:1]
        for d in data:
            pp = perplexity_calcul(n=1, testset=d, model=proba_matrix, list_unigram=indx_and_names)
            print pp
            pp = perplexity_calcul(n=1, testset=d, model=proba_lissage_matrix, list_unigram=indx_and_names_lissages)
            print pp
            pp = perplexity_calcul(n=1, testset=d, model=proba_interpolation_matrix, list_unigram=indx_and_names_interpolation)
            print pp


def main():
    count_n_gram()
    unigram_experience(fichier_test="listes_en_desordre.txt")

if __name__ == '__main__':
    main()