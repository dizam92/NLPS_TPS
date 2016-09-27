# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import re
def get_ingredients(item):
    assert isinstance(item, str), "the input must be a string"
    # main_pattern = re.compile(r'(\(?(\d+(\,?|\/?|\sà)?\d*)\s([éèêàûôîç\'’,.-]*\w+[éèêàûôîç\'’,.-]*\w*(\sà\scafé|\sà\ssoupe|\sà\sthé|\sà (.s|s.|c.))?)\)?)')
    main_pattern = re.compile(r'(\(?(\d+(\,?|\/?|\sà)?\d*)\s([éèêàûôîç\'’,.-]*\w+[éèêàûôîç\'’,.-]*\w*(\sà\scafé|\sà\ssoupe|\sà\sthé|\sà (.s|s.|c.))?)\)?)(\s.[^\(\)]+)', re.U|re.M|re.I)
    # main_pattern = re.compile(
    #     r'(\(?(\d+(\,?|\/?|\sà)?\d*)\s(\w+(\sà\scafé|\sà\ssoupe|\sà\sthé|\sà (.s|s.|c.))?)\)?)(\s.[^\(\)]+)',
    #     re.U | re.M | re.I)
    matches = main_pattern.match(item)
    if matches:
        print (matches.group(1), matches.group(7))
    else:
        print (" ", item)

def main():
    with open("liste_ingredients.txt", "r") as fichier:
        lignes = fichier.readlines()
        for l in lignes:
            print l
            l = l[:-1]
            get_ingredients(l)

if __name__ == '__main__':
    main()
    # get_ingredients("45 ml (3 c. à soupe) d’huile d’olive")