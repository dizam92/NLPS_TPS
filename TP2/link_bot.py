# -*- coding: utf-8 -*-
import argparse, os, time
import urlparse, random
import os, random, sys, time, urlparse
from selenium import webdriver
from bs4 import BeautifulSoup
import codecs
import copy
from subprocess import Popen
download_dir = "/home/prtos/Downloads/resumes_linkedin"
results_dir = "results"
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


def LInBot(browser):
    # the list of people we have visited
    visited = {}
    # all people we plan to visit
    pList = []
    # count
    count = 0
    outfile = codecs.open('Resumes_IDs.txt', 'a', 'utf8')
    while True and count <= 6000:
        count += 1
        # sleep to make sure everything loads, add random to make us look human.
        time.sleep(random.uniform(1, 4))
        # source of current brow
        page = BeautifulSoup(browser.page_source)
        # print 'page source', page
        # people is a list
        people = getPeopleLinks(page)
        # print 'people', people

        # print np.shape(people)
        if people:
            for person in people:
                ID = getID(person)
                if ID not in visited:
                    pList.append(person)
                    visited[ID] = 1
        if pList:  # if there is people to look at look at them people to visit
            # the person on the top of the list
            person = pList.pop()
            browser.get(person)
            page = BeautifulSoup(browser.page_source)
            resume = getResume(page)
            ID = getID(resume)
            ID = ID.decode('utf-8')
            ID = ID.encode('utf-8')
            try:
                name = getName(resume).encode('utf-8')
                correct_name = True
            except:
                continue
            print ID, name,
            if correct_name:
                before_download = os.listdir(download_dir)
                browser.get('https://www.linkedin.com/' + resume)
                time.sleep(random.uniform(5, 15))
                after_download = os.listdir(download_dir)
                change = set(after_download) - set(before_download)
                full_name = download_dir + '/' + change.pop()

                if os.path.exists(full_name):
                    print "downloaded"
                    outfile.write(ID + '\t' + name + '\n')
                    os.system("pdftotext -htmlmeta {} {}/{}.txt".format(full_name.replace(' ', '\\ '), results_dir, ID))
                os.remove(full_name)

        print "[+] " + browser.title + " Visited! \n(" \
              + str(count) + "/" + str(len(pList)) + ") Visited/Queue)"

    outfile.close()

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


def main(resume=None):
    assert isinstance(resume, bool), "Resume must be a boolean"
    if resume:
        email = "humain.learning@gmail.com"
        password = "hanaNADAL12"

        # Check if the file 'visitedUsers.txt' exists, otherwise create it
        if os.path.isfile('visitedUsers.txt') == False:
            visitedUsersFile = open('visitedUsers.txt', 'wb')
            visitedUsersFile.close()

        # options = webdriver.ChromeOptions()
        # options.add_experimental_option('prefs', {'download.default_directory': download_dir})
        print '\nLaunching Chrome'
        browser = webdriver.Chrome()

        # Sign in
        browser.get('https://linkedin.com/uas/login')
        emailElement = browser.find_element_by_id('session_key-login')
        emailElement.send_keys(email)
        passElement = browser.find_element_by_id('session_password-login')
        passElement.send_keys(password)
        passElement.submit()

        print 'Signing in...'
        time.sleep(3)

        soup = BeautifulSoup(browser.page_source)
        if soup.find('div', {'class': 'alert error'}):
            print 'Error! Please verify your username and password.'
            browser.quit()
        elif browser.title == '403: Forbidden':
            print 'LinkedIn is momentarily unavailable. Please wait a moment, then try again.'
            browser.quit()
        else:
            print 'Success!\n'
            LInBot(browser)

        browser.close()

    else:
        email = "kartelpalmer957@gmail.com"
        password = "ulaval16"

        print '\nLaunching Chrome'
        browser = webdriver.Chrome()
        # Sign in

        browser.get('https://linkedin.com/uas/login')
        emailElement = browser.find_element_by_id('session_key-login')
        emailElement.send_keys(email)
        passElement = browser.find_element_by_id('session_password-login')
        passElement.send_keys(password)
        passElement.submit()
        time.sleep(random.uniform(10, 20))
        os.system('clear')
        print "[+] Success! Logged In, Bot Starting!"
        job_parser(browser)
        browser.close()

if __name__ == '__main__':
    main(resume=False)
