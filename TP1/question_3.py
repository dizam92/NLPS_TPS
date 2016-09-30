# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import pandas as pd
import csv
import re
import ast
from collections import defaultdict
from time import time

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
            d.insert(0, "<s>")
            d.insert(-1, "</s>")
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
            print d
            d.insert(0, "<s>")
            d.insert(-1, "</s>")
            d.insert(0, "<s>")
            d.insert(-1, "</s>")
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
    """ New idea: Build a datastructure to keep the trigram, bigram, and unigram together """
    final_dictionnary = defaultdict(list)
    unigram_count_dictionnary = defaultdict(int)
    bigram_count_dictionnary = defaultdict(int)
    trigram_count_dictionnary = defaultdict(int)
    number_of_token = 0
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        lignes = lignes[:10]
        print len(lignes)
        data = [line[:-1].split(" ") for line in lignes]
        # data = data[:10]
        # print data
        for d in data:
            print d
            number_of_token += len(d)
            d.insert(0, "<s>")
            d.insert(-1, "</s>")
            d.insert(0, "<s>")
            d.insert(-1, "</s>")
            for i in xrange(len(d) - 2):
                uni_key = d[i]
                bin_key = d[i] + " " + d[i + 1]
                tri_key = d[i] + " " + d[i + 1] + " " + d[i + 2]
                # print bin_key
                # print tri_key
                if uni_key not in unigram_count_dictionnary.keys():
                    unigram_count_dictionnary[uni_key] = 1
                if uni_key in unigram_count_dictionnary.keys():
                    unigram_count_dictionnary[uni_key] += 1
                if bin_key not in bigram_count_dictionnary.keys():
                    bigram_count_dictionnary[bin_key] = 1
                if bin_key in bigram_count_dictionnary.keys():
                    bigram_count_dictionnary[bin_key] += 1
                if tri_key not in trigram_count_dictionnary.keys():
                    trigram_count_dictionnary[tri_key] = 1
                if tri_key in trigram_count_dictionnary.keys():
                    trigram_count_dictionnary[tri_key] += 1
    with open('unigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, unigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(unigram_count_dictionnary)
    with open('bigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, bigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(bigram_count_dictionnary)
    with open('trigram_count_dictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, trigram_count_dictionnary.keys())
        w.writeheader()
        w.writerow(trigram_count_dictionnary)

    # special_caracters = ['(', ')', '/', '?', '.', '^', '$', '*', '+', '?', '[', ']', '{', '}', '-', '+0000']
    for uni_key in unigram_count_dictionnary.keys():
        # print uni_key
        list_bin = []
        list_tri = []
        for bin_key in bigram_count_dictionnary.keys():
            b = bin_key.split(" ")
            if uni_key == b[0] or uni_key == b[1]:
                list_bin.append(bin_key)
        for tri_key in trigram_count_dictionnary.keys():
            t = tri_key.split(" ")
            if uni_key == t[0] or uni_key == t[1] or uni_key == t[2]:
                list_tri.append(tri_key)
        # for bin_key in bigram_count_dictionnary.keys():
        #     if uni_key in special_caracters:
        #         # print uni_key
        #         uni_key = "\%s" % uni_key
        #         # print uni_key
        #         matches = re.search(uni_key, bin_key, re.U | re.M | re.I)
        #     else:
        #         matches = re.search(uni_key, bin_key, re.U | re.M | re.I)
        #     if matches:
        #         list_bin.append(bin_key)
        # # print list_bin
        # for tri_key in trigram_count_dictionnary.keys():
        #     if uni_key in special_caracters:
        #         uni_key = "\%s" % uni_key
        #         matches = re.search(uni_key, bin_key, re.U | re.M | re.I)
        #     else:
        #         matches = re.search(uni_key, tri_key, re.U | re.M | re.I)
        #     if matches:
        #         list_tri.append(tri_key)
        final_dictionnary[uni_key] = [list_bin, list_tri]

    # write dictionnay to file
    with open('finaldictionnary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, final_dictionnary.keys())
        w.writeheader()
        w.writerow(final_dictionnary)
    # return final_dictionnary, unigram_count_dictionnary, bigram_count_dictionnary, trigram_count_dictionnary, number_of_token

def probability_builder(n, number_of_token):
    """ Build the matrix of the probability for the 1, 2, 3 gram"""
    if n == 1:
        d = pd.read_csv("unigram_count_dictionnary.csv")
        words = d.columns.values
        unigram_probability = d.values / float(number_of_token)
        return words, unigram_probability

    # Explications: Construire une matrice carré de words² elements. Pour chaque mot (Vu que c'est trié c'est dans le meme ordre)
    # regarder si le mot se retrouve au niveau w_(i-1) du bigram soit la position 2 dans le bigram,
    # si oui, on calcule le c(w_i, w_(i-1))/c(w_i) et mettre dans la matrice à la position (w_(i-1), w_i)
    if n == 2:
        linked_words = pd.read_csv("finaldictionnary.csv")
        bigram_proba_arrays = np.zeros((linked_words.columns.size, linked_words.columns.size), dtype=float)
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        for index, cle in enumerate(np.sort(linked_words.columns.values)):
            print "keys is ", cle
            bigram_list = ast.literal_eval(linked_words[cle][0])[0]  # get the list of the bigram only
            for el in bigram_list:
                words = el.split(" ")
                if words[1] == cle: # look if 2nd word is key
                    print el
                    print data_bi[el] / float(data_uni[words[0]])
                    bigram_proba_arrays[index, np.where(linked_words.columns.values == words[0])[0][0]] = data_bi[el] / float(data_uni[words[0]])
        print bigram_proba_arrays
        return bigram_proba_arrays, np.sort(linked_words.columns.values)

    if n == 3:
        linked_words = pd.read_csv("finaldictionnary.csv")
        data_tri = pd.read_csv("trigram_count_dictionnary.csv")
        data_bi = pd.read_csv("bigram_count_dictionnary.csv")
        # data_uni = pd.read_csv("unigram_count_dictionnary.csv")
        trigram_proba_arrays = np.zeros((linked_words.columns.size, data_bi.columns.size), dtype=float)
        for index, cle in enumerate(np.sort(linked_words.columns.values)[:10]):
            trigram_list = ast.literal_eval(linked_words[cle][0])[1]  # get the list of the bigram only
            for el in trigram_list:
                words = el.split(" ")
                if words[2] == cle:
                    trigram_proba_arrays[index, np.where(np.sort(data_bi.columns.values) == words[words[0] + " " + words[1]])[0][0]] = \
                        data_tri[el] / float(data_bi[words[0] + " " + words[1]])
        return trigram_proba_arrays, np.sort(linked_words.columns.values), np.sort(data_bi.columns.values)

def main():
    count_1_gram()
    exit()
    count_n_gram()
    exit()
    number_of_token = 261267
    proba_matrix, names = probability_builder(2, number_of_token)
    print np.where(proba_matrix != 0.0)


if __name__ == '__main__':
    main()