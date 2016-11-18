import urllib2
from  BeautifulSoup import BeautifulSoup
import re
import codecs
import numpy as np

opener = urllib2.build_opener()
opener.addheaders = [('User-agent','Google Chrome')]
url = ('https://en.wikipedia.org/wiki/List_of_American_comedy_films')
ourUrl = opener.open(url).read()
soup = BeautifulSoup(ourUrl)
i = 0
files = []
outfile = codecs.open('wiki_movies.txt', 'w', 'utf8')
for link in soup.findAll('a', attrs = {'href': re.compile("^/wiki/")}):
    i+=1
    a = link.text
    a = a.encode('utf-8')
    files.append(a)

files = [line for line in files if '!' not in line]
print type(files)
print i

print files
for i in range(1,2364):
    print files[i] 
    files[i].encode('utf-8')
    outfile.write(files[i])
    
outfile.close()


