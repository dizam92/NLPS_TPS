# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import re
from collections import defaultdict
from time import time

def count_1_gram():
    dictionnaire = defaultdict(int)
    unigram_probability_dictionnary = defaultdict(int)
    number_of_token = 0
    pattern_unigram = r'\w+|[‘\.,;:\?!\'\/’–-]'
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        data = [line[:-1].split(" ") for line in lignes]
        start_time = time()
        for d in data:
            for el in d:
                if el not in dictionnaire.keys():
                    dictionnaire[el] = 1
                else:
                    dictionnaire[el] += 1
        end_time = time() - start_time
        print (end_time)
        start_time = time()
        for l in lignes:
            l = l[:-1]
            matches = re.findall(pattern_unigram, l, re.U | re.M | re.I)
            number_of_token += len(matches)
            for el in matches:
                if el not in dictionnaire.keys():
                    dictionnaire[el] = 1
                else:
                    dictionnaire[el] += 1
        end_time = time() - start_time
        print (end_time)
    for k in dictionnaire.keys():
        proba = dictionnaire[k] / float(number_of_token)
        unigram_probability_dictionnary[k] = proba
    # with open("unigrams_count.txt", "w") as fichier:
    #     fichier.write("Number of tokens in this corpus is; %d\n" % number_of_token)
    #     for k in dictionnaire.keys():
    #         proba = dictionnaire[k]/float(number_of_token)
    #         unigram_probability_dictionnary[k] = proba
    #         fichier.write("%s: %d Probality: %f\n" % (k, dictionnaire[k], proba))
    return unigram_probability_dictionnary

def count_2_gram():
    dictionnaire = defaultdict(int)
    unigram_dictionnary_count = defaultdict(int)
    number_of_token = 0
    pattern_bigram = r'\w+|[‘\.,;:\?!\'\/’–-]'
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        for l in lignes:
            l = l[:-1]
            matches = re.findall(pattern_bigram, l, re.U | re.M | re.I)
            number_of_token += len(matches)
            # consider the beginning and ending phrase in our global count????
            matches.insert(0, "<s>")
            matches.insert(-1, "</s>")
            for i in xrange(len(matches)-1):
                bin_key = matches[i]+" "+matches[i+1]
                if bin_key not in dictionnaire.keys():
                    dictionnaire[bin_key] = 1
                else:
                    dictionnaire[bin_key] += 1

            for el in matches:
                if el not in unigram_dictionnary_count.keys():
                    unigram_dictionnary_count[el] = 1
                else:
                    unigram_dictionnary_count[el] += 1
    # with open("bigrams_count.txt", "w") as fichier:
    #     fichier.write("Number of tokens in this corpus is; %d\n" % number_of_token)
    #     for k in dictionnaire.keys():
    #         fichier.write("%s: %d\n" % (k, dictionnaire[k]))
    return unigram_dictionnary_count, dictionnaire

def count_3_gram():
    dictionnaire = defaultdict(int)
    bigram_dictionnary_count = defaultdict(int)
    number_of_token = 0
    pattern_trigram = r'\w+|[‘\.,;:\?!\'\/’–-]'
    with open("corpus_small.txt", "r") as fichier:
        lignes = fichier.readlines()
        for l in lignes:
            l = l[:-1]
            matches = re.findall(pattern_trigram, l, re.U | re.M | re.I)
            number_of_token += len(matches)
            # consider the beginning and ending phrase in our global count????
            matches.insert(0, "<s>")
            matches.insert(0, "<s>")
            matches.insert(-1, "</s>")
            matches.insert(-1, "</s>")
            for i in xrange(len(matches) - 1):
                tri_key = matches[i] + " " + matches[i + 1] + " " + matches[i + 2]
                if tri_key not in dictionnaire.keys():
                    dictionnaire[tri_key] = 1
                else:
                    dictionnaire[tri_key] += 1

            for i in xrange(len(matches) - 1):
                bin_key = matches[i] + " " + matches[i + 1]
                if bin_key not in dictionnaire.keys():
                    bigram_dictionnary_count[bin_key] = 1
                else:
                    bigram_dictionnary_count[bin_key] += 1
    # with open("trigrams_count.txt", "w") as fichier:
    #     fichier.write("Number of tokens in this corpus is; %d\n" % number_of_token)
    #     for k in dictionnaire.keys():
    #         fichier.write("%s: %d\n" % (k, dictionnaire[k]))
    return bigram_dictionnary_count, dictionnaire

def count_ngram(n):
    """ Compute the count for each type of word (unique word) """
    assert n == 1 or n ==2 or n == 4, "The ngram you want is not implemented right now"
    if n == 1:
        unigram_dict_probability = count_1_gram()
    if n == 2:
        unigram_dict_count, bigram_dict = count_2_gram()
    if n == 3:
        bigram_dict_count, trigram_dict = count_3_gram()

def main():
    # to stock the information, but dont know right now if it's worth on a text file
    # with open("unigrams_count.txt", "w") as fichier:
    #     fichier.write("Number of tokens in this corpus is; %d\n" % number_of_token)
    #     for k in dictionnaire.keys():
    #         proba = dictionnaire[k]/float(number_of_token)
    #         dictionnaire_de_probabilite[k] = proba
    #         fichier.write("%s: %d Probality: %f\n" % (k, dictionnaire[k], proba))

    # to stock the information, but dont know right now if it's worth on a text file
    # with open("bigrams_count.txt", "w") as fichier:
    #     fichier.write("Number of tokens in this corpus is; %d\n" % number_of_token)
    #     for k in dictionnaire.keys():
    #         proba = dictionnaire[k]/float(number_of_token)
    #         bigram_probability_dictionnary[k] = proba
    #         fichier.write("%s: %d Probality: %f\n" % (k, dictionnaire[k], proba))
    count_ngram(1)

if __name__ == '__main__':
    main()