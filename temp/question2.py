import re
import numpy as np
from sklearn.metrics import confusion_matrix

question_to_classes = {"QUANTITY": 0, "LOCATION": 1, "TEMPORAL": 2, "DEFINITION": 3, "PERSON": 4}


def recognize(pattern, texte):
    if re.search(pattern, texte):
        return True
    else:
        return False


def predict_question_type(texte):
    pattern_quantity = "((worth)|([hH]ow ((far)|(tall)|(much)|(many)|(fast)|(cold)|(wide)|(often)|(long) (?!((did)|(is)))))|([Ww]hat is the .*((width)|(diameter)|(length)|(distance)|(rate)|(MO)|(weight)|(depth)|(population)|(percentage)|(temperature)|(speed)|(expectancy)|(melting point))))"
    pattern_location = "(((([Ww]hat)|([wW]hich)) .*((place)|(state)|(strait)|(river)|(street)|(lake)|(continent)|(mountain)|(territories)|(seaport)|(part of)|(colony)|(bridge)|(capital)|(country)|(county)|([cC]it(y|(ies)))|(province)|(planet)|(star)|(location)|(hemisphere)|( line)))|([wW]here))"
    pattern_temporal = "(When)|([wW]hat .*(((is .*)?date )|(year)|(day)|(month)|(period)|(span)))|(How old)|((For )?[Hh]ow long)|(During which)"
    pattern_definition = "([wW]hat.?((is (?!the))|(are (?!the))|(do(es)?)))|(Why)|(How ((do(es)?)|(did)))"
    pattern_person = "(Who)|(What .*((name)|(composer)|(person)|(ruler)))|(Which .*((president)|(person)|(comedian)))"

    if recognize(pattern_person, texte):
        return question_to_classes["PERSON"]
    if recognize(pattern_quantity, texte):
        return question_to_classes["QUANTITY"]
    if recognize(pattern_location, texte):
        return question_to_classes["LOCATION"]
    if recognize(pattern_temporal, texte):
        return question_to_classes["TEMPORAL"]
    if recognize(pattern_definition, texte):
        return question_to_classes["DEFINITION"]
    return question_to_classes["DEFINITION"]
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
        tp_cl1 = len(np.where(p[answers == 0] == 0)[0])
        tp_cl2 = len(np.where(p[answers == 1] == 1)[0])
        tp_cl3 = len(np.where(p[answers == 2] == 2)[0])
        tp_cl4 = len(np.where(p[answers == 3] == 3)[0])
        tp_cl5 = len(np.where(p[answers == 4] == 4)[0])
        tp = tp_cl1+tp_cl2+tp_cl3+tp_cl4+tp_cl5

        fp_cl1 = len(np.where(p[answers != 0] == 0)[0])
        fp_cl2 = len(np.where(p[answers != 1] == 1)[0])
        fp_cl3 = len(np.where(p[answers != 2] == 2)[0])
        fp_cl4 = len(np.where(p[answers != 3] == 3)[0])
        fp_cl5 = len(np.where(p[answers != 4] == 4)[0])
        fp = fp_cl1 + fp_cl2 + fp_cl3 + fp_cl4 + fp_cl5

        fn = len(np.where(p[answers == 1] == 0)[0])

        fn_cl1 = len(np.where(p[answers == 0] != 0)[0])
        fn_cl2 = len(np.where(p[answers == 1] != 1)[0])
        fn_cl3 = len(np.where(p[answers == 2] != 2)[0])
        fn_cl4 = len(np.where(p[answers == 3] != 3)[0])
        fn_cl5 = len(np.where(p[answers == 4] != 4)[0])
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

if __name__ == "__main__":
    with open("questions.txt", "r") as f:
        all_targets = []
        all_preds = []
        for ligne in f.readlines():
            temp = ligne.split()
            target, input = question_to_classes[temp[0]], " ".join(temp[1:])
            pred = predict_question_type(ligne[:-1])
            all_targets += [target]
            all_preds += [pred]

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        metrics = _get_metrics(all_preds, all_targets)
        print metrics
        metrics = confusion_matrix(y_true=all_targets, y_pred=all_preds)
        print metrics
        print "Accuracy: ", np.mean(all_targets == all_preds)

        for nom, classe in question_to_classes.iteritems():
            acc = np.mean(all_targets[all_targets == classe] == all_preds[all_targets == classe])
            print nom, acc

        print "{:12}".format(""),
        for nom, _ in question_to_classes.iteritems():
            print "{:12}".format(nom),
        print
        for nom1, classe1 in question_to_classes.iteritems():
            print "{:12}".format(nom1),
            for nom2, classe2 in question_to_classes.iteritems():
                acc = np.sum(all_targets[all_targets == classe1] == all_preds[all_targets == classe2])
                print "{:12}".format(acc),
            print




