# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import numpy as np
import re

def get_types(line):
    """ Return the type of the line"""
    pattern_Location = r"(Where)|" \
                       "(Which) (state|country)|" \
                       "(What|What is the) (\w+est|\w+al|street|location|tower|river|tunnel|Mal|city|cities|continent|canada|world|sea|Lake|country|house|star|planet|U\.S\.|French|state|asia|county|mountain)"

    pattern_Quantity = r"(How) (far|much|fast|tall|wide|many|often)|" \
                       "((What is the) (population|temperature|distance|average|percentage|life|width|length|diameter|speed|melting point|sales|earth))"

    pattern_Temporal = r"(How) (long did|old)|" \
                       "(when)|" \
                       "(what|which) (year|day|period|date|season|was the)"

    pattern_Definition = r"(How) (do|did|does)|" \
                         "(Why)|" \
                         "(what) (does|are|is (?!the)|is a|is an)"

    pattern_Person = r"(Who) (is|was|won|\w+[ed])|" \
                     "(What|Which) (comedian|president|person|name)"

    list_of_pattern = [pattern_Quantity, pattern_Location, pattern_Definition, pattern_Temporal, pattern_Person]
    temp_list = np.zeros((5,), dtype=int)
    for index, pattern in enumerate(list_of_pattern):
        matches = re.search(pattern, line, re.U|re.M|re.I)
        if matches:
            temp_list[index] = 1
    temp_true = np.where(temp_list==1)
    if temp_true[0].size:
        return temp_true[0][0]+1
    else:
        return 0

def create_label(word):
    assert isinstance(word, str), "the input must be a string"
    if word == "QUANTITY":
        return 1
    elif word == "LOCATION":
        return 2
    elif word == "DEFINITION":
        return 3
    elif word == "TEMPORAL":
        return 4
    elif word == "PERSON":
        return 5

def _get_metrics(predictions, answers):
    """ Compute the metrics and the risk manually.
    Interesting in case you vwant to perform yourself the metrics and report the metrics for training/testing separately
    For classification ONLY!!!
    """
    from collections import defaultdict
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
    metrics = defaultdict(list)
    for i in xrange(predictions.shape[0]):
        p = predictions[i]
        risk = 1.0 * len(p[p != answers]) / len(answers)
        tp_cl1 = len(np.where(p[answers == 1] == 1)[0])
        tp_cl2 = len(np.where(p[answers == 2] == 2)[0])
        tp_cl3 = len(np.where(p[answers == 3] == 3)[0])
        tp_cl4 = len(np.where(p[answers == 4] == 4)[0])
        tp_cl5 = len(np.where(p[answers == 5] == 5)[0])
        tp = tp_cl1+tp_cl2+tp_cl3+tp_cl4+tp_cl5

        fp_cl1 = len(np.where(p[answers != 1] == 1)[0])
        fp_cl2 = len(np.where(p[answers != 2] == 2)[0])
        fp_cl3 = len(np.where(p[answers != 3] == 3)[0])
        fp_cl4 = len(np.where(p[answers != 4] == 4)[0])
        fp_cl5 = len(np.where(p[answers != 5] == 5)[0])
        fp = fp_cl1 + fp_cl2 + fp_cl3 + fp_cl4 + fp_cl5

        fn = len(np.where(p[answers == 1] == 0)[0])

        fn_cl1 = len(np.where(p[answers == 1] != 1)[0])
        fn_cl2 = len(np.where(p[answers == 2] != 2)[0])
        fn_cl3 = len(np.where(p[answers == 3] != 3)[0])
        fn_cl4 = len(np.where(p[answers == 4] != 4)[0])
        fn_cl5 = len(np.where(p[answers == 5] != 5)[0])
        fn = fn_cl1 + fn_cl2 + fn_cl3 + fn_cl4 + fn_cl5
        precision = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else -np.infty
        sensitivity = recall = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else -np.infty
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else -np.infty
        metrics["risk"].append(risk)
        metrics["tp"].append(tp)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["precision"].append(precision)
        metrics["sensitivity"].append(sensitivity)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1_score)
    return metrics

def main():
    y_true = []
    y_pred = []
    with open("questions.txt", "r") as fichier:
        lignes = fichier.readlines()
        for l in lignes:
            print l
            l = l[:-1]
            data = [l.split(" ")]
            y_true.append(create_label(data[0][0]))
            y_pred.append(get_types(l))
        print y_true
        print y_pred
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
    metrics =_get_metrics(y_pred,y_true)
    print metrics
if __name__ == '__main__':
    main()