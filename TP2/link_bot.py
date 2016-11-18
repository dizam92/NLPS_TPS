#-*- coding: utf-8 -*-
import argparse, os, time
import urlparse, random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import numpy as np
import os
import pyPdf
import codecs
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
    #link_resume = []
    for link in page.findAll('a'):
        #print 'link resume', link
        url = link.get('href')
        if url:
            if 'pdf_pro_full' in url:
                print 'url_resume', url
                return url
    #return link_resume
    
def getJobLinks(page):
    links = []
    for link in page.findAll('a'):
        url = link.get('href')
        if url:        
            if '/jobs2' in url:
                links.append(url)
    return links

def getID(url):
    pUrl = urlparse.urlparse(url)
    return urlparse.parse_qs(pUrl.query)['id'][0]

def getName(url):
    pUrl = urlparse.urlparse(url)
    return urlparse.parse_qs(pUrl.query)['pdfFileName'][0]

def ViewBot(browser):
    # the list of people we have visited
    visited = {}
    # all people we plan to visit
    pList = []
    # count
    count = 0
    download_dir = '/home/Downloads/resumes_linkedin'
    outfile = codecs.open('Resumes_IDs.txt', 'w+', 'utf8')
    while True and count <=1500:
        count+= 1
        #sleep to make sure everything loads, add random to make us look human.
        time.sleep(random.uniform(3,6))
        # source of current brow
        page = BeautifulSoup(browser.page_source)
        #print 'page source', page
        #people is a list
        people = getPeopleLinks(page)
        #print 'people', people
        
        #print np.shape(people)
        if people:
            for person in people:
                ID = getID(person)
                if ID not in visited:
                    pList.append(person)
                    visited[ID] = 1
        if pList: #if there is people to look at look at them people to visit
            # the person on the top of the list
            person = pList.pop()
            print 'person', person
            browser.get(person)
            page = BeautifulSoup(browser.page_source)
            print 'get to person'
            resume = getResume(page)
            ID = getID(resume)
            ID = ID.decode('utf-8')
            ID  = ID.encode('utf-8')
            print type(ID)
            Name = getName(resume).encode('utf-8')
            #print 'resume',ID, Name
            browser.get('https://www.linkedin.com/'+resume)
            time.sleep(random.uniform(20,45))
            if os.path.exists(download_dir+'/'+Name+'.pdf'):
                outfile.write(ID+ '\t' + Name + '\n')
                new_name = download_dir+'/'+str(ID)+'.pdf'
                os.rename(download_dir+'/'+Name+'.pdf' , new_name)
                Popen("pdftotext "+ new_name, shell=True)
            
        
        print "[+] "+browser.title+" Visited! \n("\
            +str(count)+"/"+str(len(pList))+") Visited/Queue)"
            
    outfile.close()
     
                    

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("email", help="linkedin email")
    parser.add_argument("password", help="linkedin password")
    args = parser.parse_args()
    # on cree un browser firefox souvre
    download_dir = '/home/Downloads/resumes_linkedin'
    # profile = webdriver.FirefoxProfile()
    # profile = webdriver.Chrome
    # profile.set_preference("browser.download.folderList", 2)
    # profile.set_preference("browser.download.manager.showWhenStarting", False)
    # profile.set_preference("browser.download.dir", download_dir)
    # profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    #
    # profile.set_preference("pdfjs.disabled", True)
    # profile.set_preference("plugin.scan.plid.all", False)
    # profile.set_preference("plugin.scan.Acrobat", "99.0")

    # browser = webdriver.Chrome(executable_path="/usr/bin/google-chrome")
    # browser = webdriver.Firefox(executable_path='/home/maoss2/firefox/firefox-bin')
    browser = webdriver.Firefox()
    # on se connecte a la page de connexion linkdin
    browser.get("https://linkedin.com/uas/login")
    
   
    # on trouve lendroit ou mettre l email et le passeword
    emailElement = browser.find_element_by_id("session_key-login")
    emailElement.send_keys(args.email)
    passElement = browser.find_element_by_id("session_password-login")
    passElement.send_keys(args.password)
    # a submit form
    passElement.submit()

    os.system('clear')
    print "[+] Success! Logged In, Bot Starting!"
    # the viewbot browser
    ViewBot(browser)
    #browser.close()

if __name__ == '__main__':
    Main()
