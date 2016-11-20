# -*- coding: utf-8 -*-
import argparse, os, time
import urlparse, random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import numpy as np
import os
import pyPdf
import codecs
import copy
from subprocess import Popen


def getPeopleLinks(page):
    links = []
    # the a tags
    for link in page.findAll('a'):
        url = link.get('href')
        if url:
            if 'profile/view?id=' in url:
                links.append(url)
    return links


def getResume(page):
    # link_resume = []
    for link in page.findAll('a'):
        # print 'link resume', link
        url = link.get('href')
        if url:
            if 'pdf_pro_full' in url:
                print 'url_resume', url
                return url
                # return link_resume

def getJobLinks(page):
    links = []
    for link in page.findAll('a'):
        url = link.get('href')
        if url:
            if 'jobs?viewJob' in url:
                links.append(url)
    return links


def getJobLinks2(browser):
    links = []
    d = browser.find_element_by_class_name("compact-jobs-container")
    all_items = d.find_elements_by_class_name("item")
    for items in all_items:
        links.append(items.find_element_by_css_selector("a").get_attribute("href"))
    return links

def getJobID(url):
    pUrl = urlparse.urlparse(url)
    if url.find('jobId') > 0:
        return urlparse.parse_qs(pUrl.query)['jobId'][0]
    else:
        return url[35:44]


def getID(url):
    pUrl = urlparse.urlparse(url)
    return urlparse.parse_qs(pUrl.query)['id'][0]


def getName(url):
    pUrl = urlparse.urlparse(url)
    return urlparse.parse_qs(pUrl.query)['pdfFileName'][0]


def viewBot_job(browser):
    # the list of people we have visited
    visited = {}
    # all people we plan to visit
    pList = []
    # count
    count = 0
    download_dir = '/home/Downloads/resumes_linkedin'
    outfile = codecs.open('Resumes_IDs.txt', 'w+', 'utf8')
    while True and count <= 1500:
        count += 1
        # sleep to make sure everything loads, add random to make us look human.
        time.sleep(random.uniform(3, 6))
        # source of current browser
        page = BeautifulSoup(browser.page_source)
        jobs = getJobLinks(page)
        browser.get(jobs[0])
        browser.find_element_by_id("job-details-reveal").click()
        el = browser.find_element_by_class_name('posting-content').text
        # els = browser.find_elements_by_class_name('description-section')
        # for el in els:
        #     print el.text
        print el
        exit()
        # job1_page.find('h3')
        description = BeautifulSoup(browser.page_source)
        s = description.find_('div')
        print s
        exit()
        # print 'people', people
        if jobs:
            for el in jobs:
                browser.get(el)
                # print el
                # ID = getJobID(el)
                # ID = ID.decode('utf-8')
                # ID = ID.encode('utf-8')
                # print ID
                # print type(ID)
                # browser.get(el)
                # time.sleep(random.uniform(20, 45))
                # if os.path.exists(download_dir + '/' + ID + '.pdf'):
                #     print "true"
                #     outfile.write(ID + '\n')
                #     new_name = download_dir + '/' + str(ID) + '.pdf'
                #     os.rename(download_dir + '/' + ID + '.pdf', new_name)
                #     Popen("pdftotext " + new_name, shell=True)
                exit()


def job_parser(browser):
    # the list of people we have visited
    visited = {}
    # all people we plan to visit
    pList = []
    # count
    count = 0
    download_dir = '/home/maoss2/PycharmProjects/NLP_TPS/TP2'
    outfile = codecs.open('Resumes_IDs.txt', 'a+', 'utf8')

    page = BeautifulSoup(browser.page_source)
    jobs = getJobLinks(page)
    print jobs
    browser.get(jobs[0])

    while True and count <= 20:
        count += 1
        # sleep to make sure everything loads, add random to make us look human.
        time.sleep(random.uniform(3, 6))
        # source of current browser
        jobs = getJobLinks2(browser)
        print jobs
        if jobs:
            for el in jobs:
                ID = getJobID(el)
                if ID not in visited:
                    pList.append(el)
                    visited[ID] = 1
        print "plist len", len(pList)
        #     for el in pList:
        #         browser.get(el)
        #         d = browser.find_element_by_class_name("compact-jobs-container")
        #         all_items = d.find_elements_by_class_name("item")
        #         for items in all_items:
        #             jList.append(items.find_element_by_css_selector("a").get_attribute("href"))
        # pList = pList + jList
        if pList:  # if there is job to look
            # the job on the top of the list
            job = pList.pop()
            browser.get(job)
            time.sleep(random.uniform(5, 15))
            browser.find_element_by_id("job-details-reveal").click()
            CV = browser.find_element_by_class_name('posting-content').text
            ID = getJobID(job)
            ID = ID.decode('utf-8')
            ID = ID.encode('utf-8')
            print ID
            # print CV
            time.sleep(random.uniform(0, 5))
            if os.path.exists(download_dir + "job_%s.txt" % ID):
                pass
            else:
                with open("job_%s.txt" % ID, "w") as f:
                    f.write(CV.encode('utf-8'))


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("email", help="linkedin email")
    parser.add_argument("password", help="linkedin password")
    args = parser.parse_args()
    # on cree un browser firefox souvre
    download_dir = '/home/Downloads/resumes_linkedin'

    browser = webdriver.Chrome()
    # on se connecte a la page de connexion linkdin
    browser.get("https://linkedin.com/uas/login")

    # on trouve lendroit ou mettre l email et le passeword

    emailElement = browser.find_element_by_id("session_key-login")
    emailElement.send_keys(args.email)
    passElement = browser.find_element_by_id("session_password-login")
    passElement.send_keys(args.password)
    # a submit form
    passElement.submit()
    time.sleep(random.uniform(10, 20))
    os.system('clear')
    print "[+] Success! Logged In, Bot Starting!"
    # the viewbot browser
    # ViewBot(browser)
    # browser.close()
    # viewBot_job(browser)
    job_parser(browser)

if __name__ == '__main__':
    Main()
