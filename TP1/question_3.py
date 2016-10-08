# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import pandas as pd
import csv
from itertools import permutations
from collections import defaultdict
import random
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
    # print "Linking the n gram keys......"
    # linked_dictionnary_builder(unigram_file="unigram_count_dictionnary.csv",
    #                            bigram_file="bigram_count_dictionnary.csv",
    #                            trigram_file="trigram_count_dictionnary.csv")

def unigram_model(list_of_words, unigram_count, N=count_token()):
    """ Compute the MLE for unigram model
    Params: list_of_words: list of the unique word to be learned
            unigram_count: dictionnary of the count of the words in the corpus"""
    d = pd.read_csv(unigram_count)
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
    temp_list = []
    for i in xrange(len(list_of_words) - 2):
        # print list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]
        # trigram in dictionnary and bigram exists too

        if list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2] in trigram_count.columns.values\
                and list_of_words[i] + " " + list_of_words[i + 1] in bigram_count.columns.values:
            temp_list.append((list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2],
            ((trigram_count[list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]].values[0] + delta) /
            float(bigram_count[list_of_words[i] + " " + list_of_words[i + 1]].values[0] + uni_count.columns.values.size))))

        # trigram in dictionnary and bigram doesnt exists
        elif list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2] in trigram_count.columns.values\
                and list_of_words[i] + " " + list_of_words[i + 1] not in bigram_count.columns.values:
            temp_list.append((list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2],
            ((trigram_count[list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2]].values[0] + delta) /
            uni_count.columns.values.size)))

        else:
            temp_list.append((list_of_words[i] + " " + list_of_words[i + 1] + " " + list_of_words[i + 2],
                              (delta / float(uni_count.columns.values.size))))
    proba_dict = dict(temp_list)
    return proba_dict

def interpolation_trigram_model(list_of_words, unigram_count, bigram_count, trigram_count, N=count_token(), lambda1=None, lambda2=None, lambda3=None):
    """ Apply the interpolation lissage"""

    # A modifier
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

def perplexity_unigram_model(test_set, model_probability_of_the_test_set):
    log_proba = 0
    N = len(test_set)
    for el in test_set:
        log_proba += np.log10(model_probability_of_the_test_set[el])
    return np.exp(- log_proba / float(N))

def perplexity_bigram_model(test_set, model_probability_of_the_test_set):
    log_proba = 0
    N = len(test_set)
    for i in xrange(len(test_set) - 1):
        if model_probability_of_the_test_set[test_set[i] + " " + test_set[i+1]] != 0:
            log_proba += np.log10(model_probability_of_the_test_set[test_set[i] + " " + test_set[i+1]])
    return np.exp(- log_proba / float(N))

def perplexity_trigram_model(test_set, model_probability_of_the_test_set):
    log_proba = 0
    N = len(test_set)
    for i in xrange(len(test_set) - 2):
        if model_probability_of_the_test_set[test_set[i] + " " + test_set[i + 1] + " " + test_set[i + 2]] != 0:
            log_proba += np.log10(model_probability_of_the_test_set[test_set[i] + " " + test_set[i + 1] + " " + test_set[i + 2]])
    return np.exp(- log_proba / float(N))

def generate_sentence(ligne):
    """ Take a line, split it in words and return all the possible combinaison into a list of list"""
    return [subset for subset in permutations(ligne, len(ligne))]

def bigram_experiments(test_file, number_of_combinaison=20, laplace=None, interpolation=None):
    if laplace is None and interpolation is None:
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                # print sentences
                for s in sentences:
                    # print s
                    model_bi = bigram_model(list_of_words=list(s), unigram_count="unigram_count_dictionnary.csv",
                                        bigram_count="bigram_count_dictionnary.csv")
                    if perplexity_bigram_model(test_set=list(s), model_probability_of_the_test_set=model_bi) < pp:
                        pp = perplexity_bigram_model(test_set=list(s), model_probability_of_the_test_set=model_bi)
                        best_sentence["sentence"] = " ".join(list(s))
                        best_sentence["pp"] = pp
                print best_sentence

    elif laplace:
        delta_list = [0.1, 0.002, 0.5, 0.01, 0.05, 0.8]
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp, "delta": 0}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                # print sentences
                for s in sentences:
                    for delta in delta_list:
                        model_bi = laplace_delta_bigram_model(list_of_words=list(s),
                                                              unigram_count="unigram_count_dictionnary.csv",
                                                              bigram_count="bigram_count_dictionnary.csv",
                                                              delta=delta)
                        if perplexity_bigram_model(test_set=list(s), model_probability_of_the_test_set=model_bi) < pp:
                            pp = perplexity_bigram_model(test_set=list(s), model_probability_of_the_test_set=model_bi)
                            best_sentence["sentence"] = " ".join(list(s))
                            best_sentence["pp"] = pp
                            best_sentence["delta"] = delta
                print best_sentence

    elif interpolation:
        lambda2 = [0.002, 0.5, 0.01]
        lambda3 = [0.1, 0.5, 0.8]
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp, "l2": 0, "l3": 0}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                # print sentences
                for s in sentences:
                    for l2 in lambda2:
                        for l3 in lambda3:
                            model_bi_interpol = interpolation_bigram_model(list_of_words=list(s),
                                                                           unigram_count="unigram_count_dictionnary.csv",
                                                                           bigram_count="bigram_count_dictionnary.csv",
                                                                           lambda2=l2, lambda3=l3)
                            if perplexity_bigram_model(test_set=list(s),
                                                       model_probability_of_the_test_set=model_bi_interpol) < pp:
                                pp = perplexity_bigram_model(test_set=list(s),
                                                             model_probability_of_the_test_set=model_bi_interpol)
                                best_sentence["sentence"] = " ".join(list(s))
                                best_sentence["pp"] = pp
                                best_sentence["l2"] = l2
                                best_sentence["l3"] = l3
                print best_sentence

def trigram_experiments(test_file, number_of_combinaison=2, laplace=None, interpolation=None):
    if laplace is None and interpolation is None:
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                # print sentences
                for s in sentences:
                    model_tri = trigram_model(list_of_words=list(s), bigram_count="bigram_count_dictionnary.csv", trigram_count="trigram_count_dictionnary.csv")
                    if perplexity_trigram_model(test_set=list(s), model_probability_of_the_test_set=model_tri) < pp:
                        pp = perplexity_trigram_model(test_set=list(s), model_probability_of_the_test_set=model_tri)
                        best_sentence["sentence"] = " ".join(list(s))
                        best_sentence["pp"] = pp
                print best_sentence

    elif laplace:
        delta_list = [0.1, 0.002, 0.5, 0.01, 0.05, 0.8]
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp, "delta": 0}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                for s in sentences:
                    for delta in delta_list:
                        model_tri = laplace_delta_trigram_model(list_of_words=list(s),
                                                                unigram_count="unigram_count_dictionnary.csv",
                                                                bigram_count="bigram_count_dictionnary.csv",
                                                                trigram_count="trigram_count_dictionnary.csv",
                                                                delta=delta)
                        if perplexity_trigram_model(test_set=list(s), model_probability_of_the_test_set=model_tri) < pp:
                            pp = perplexity_trigram_model(test_set=list(s), model_probability_of_the_test_set=model_tri)
                            best_sentence["sentence"] = " ".join(list(s))
                            best_sentence["pp"] = pp
                            best_sentence["delta"] = delta
                print best_sentence

    elif interpolation:
        lambda1 = [0.03, 0.5]
        lambda2 = [0.2, 0.4]
        lambda3 = [0.3, 0.8]
        with open(test_file, 'r') as fichier:
            lignes = fichier.readlines()
            data = [line[:-1].split(" ") for line in lignes]
            for d in data:
                pp = 1e10000
                best_sentence = {"sentence": " ", "pp": pp, "l1": 0, "l2": 0, "l3": 0}
                sentences = generate_sentence(d)
                sentences = random.sample(sentences, number_of_combinaison)
                for s in sentences:
                    for l1 in lambda1:
                        for l2 in lambda2:
                            for l3 in lambda3:
                                model_bi_interpol = interpolation_trigram_model(list_of_words=list(s),
                                                                                unigram_count="unigram_count_dictionnary.csv",
                                                                                bigram_count="bigram_count_dictionnary.csv",
                                                                                trigram_count="trigram_count_dictionnary.csv",
                                                                                lambda1=l1, lambda2=l2, lambda3=l3)
                                if perplexity_trigram_model(test_set=list(s),
                                                            model_probability_of_the_test_set=model_bi_interpol) < pp:
                                    pp = perplexity_trigram_model(test_set=list(s),
                                                                  model_probability_of_the_test_set=model_bi_interpol)
                                    best_sentence["sentence"] = " ".join(list(s))
                                    best_sentence["pp"] = pp
                                    best_sentence["l1"] = l1
                                    best_sentence["l2"] = l2
                                    best_sentence["l3"] = l3
                print best_sentence

def main():
    # count_n_gram()
    # bigram_experiments(test_file="listes_en_desordre.txt", laplace=True)
    trigram_experiments(test_file="listes_en_desordre.txt", laplace=True)

if __name__ == '__main__':
    main()