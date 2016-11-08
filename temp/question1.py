# coding: utf-8

import re
import numpy as np


def get_ingredients(texte):
    pattern_avec_qte = "(((\(?(((\d+( à \d+)?([//,]\d+)?)|((\d )?[½,¾])|([uU]ne( quinzaine)?)|(trait)|([Qq]uelques)) *(petite? )?((trait)|(feuilles)|(rôti de)|(pincée?)|(pièces?)|(bottes?)|(pintes?)|(noix)|(gousses?)|(gallons?)|(sommités?)|(enveloppes?)|(morceaux?)|(tranches?)|(verres?( à moutarde)?)|(pincée?)|(boîtes?( de conserve)?)|(tasses?)|([bB]ouquet)|([Rr]ondelles?)|(feuilles?)|((cuillères?|c\.) *à *(soupe|café|thé|\.s|\.c|s\.|c\.))|(lb)|(oz)|([mc]?[lL])|([mkK]?[gG]))?)\)? (à )?){1,2})(((d. ?))?)((([^,\n]*)(,.*)?(?:((émincée?s?)|(frais moulu)|(râpé)|(hachée?s?)|(en purée)|(rattes?)|(dans leur coquille)|(ciselée?)|(battu)|(coupée?s? (finement)?(en biseaux)?(râpée?s?)?)|(désossées et coupées en petits morceaux)|(pelée?s?)|(écrasés)|(tranchée?))))|([^,\n]*)(,.*)?))"
    pattern_sans_qte = "([Pp]résentation)|([Pp]réparation)|(Finition)"

    prog1 = re.compile(pattern_avec_qte)
    result = prog1.match(texte)

    if result:
        groups = result.groups()
        # print groups
        ingredients = groups[52].lstrip().rstrip() if groups[52] else groups[-2].lstrip().rstrip()
        return groups[1].lstrip().rstrip(), ingredients
    else:
        prog2 = re.compile(pattern_sans_qte)
        result = prog2.match(texte)
        if result:
            return "", ""
        else:
            return "", texte.lstrip().rstrip()

if __name__ == "__main__":
    preds = []
    true_targets = []
    data = open("liste_ingredients.txt", "r")
    solution = open("ingredients_solutions.txt", "r")

    total = 0
    cpt = 0
    for ligne1, ligne2 in zip(data.readlines(), solution.readlines()):
        pred = get_ingredients(ligne1[:-1])
        preds.append(pred)
        temp = ligne2[:-1].split("QUANTITE:")[1].split("INGREDIENT:")
        qte, ingredients = temp[0], temp[1]
        qte = qte.rstrip()
        ingredients = ingredients.rstrip()
        true_targets.append((qte, ingredients))

        if pred == (qte, ingredients):
            cpt += 1
        else:
            print pred
            print (qte, ingredients)
            print pred == (qte, ingredients)
            print
        total +=1
        #print ligne2
        #print pred
        #print

    print "Accuracy: ", cpt*1.0/total

