import requests as rq
import bs4 as b
import urllib as ulib
import pandas as p
import nltk

print("You have successfully imported requests version "+rq.__version__)
print ("You have successfully imported beautifulsoup version "+b.__version__)
print ("You have successfully imported pandas version "+p.__version__)

base_url = 'https://en.wikipedia.org/wiki/Jupiter'
r = rq.get(base_url)
print(r)

soup = b.BeautifulSoup(r.text,'html5lib')
print(soup)
print()

headers = []
for url in soup.findAll("h3"):
    headers.append(url.text)
    print(url.text)

    i = len(headers) - 1
counter = 0
while counter <= i:
    if headers[counter].startswith('\n'):
        headers.pop(counter)
        counter -= 1
    counter += 1
    i = len(headers) -1
print(headers)
print()

r = rq.get(base_url)
soup = b.BeautifulSoup(r.text,'html5lib')
deet = soup.find('h3', text = headers[0]) # Search for div tags of class 'entry-content content'
para = deet.find_next_sibling('p') # Within these tags, find all p tags
print(para.get_text())
print()

r = rq.get(base_url)
soup = b.BeautifulSoup(r.text,'html5lib')
deet = soup.find('h3', text = headers[0]) # Search for div tags of class 'entry-content content'

for para in deet.find_next_siblings(): # Within these tags, find all p tags
    if para.name == "h2" or para.name == "h3":
        break
    elif para.name == "p":
        print(para.get_text())
    print()

r = rq.get(base_url)
all_para = ""
soup = b.BeautifulSoup(r.text,'html5lib')
for iteri in range(len(headers)):
    deet = soup.find('h3', text = headers[iteri]) # Search for div tags of class 'entry-content content'
    for para in deet.find_next_siblings(): # Within these tags, find all p tags
        if para.name == "h2" or para.name == "h3":
            break
        elif para.name == "p":
            all_para += para.get_text()
            all_para += '\n'
        
print(all_para)
print()

with open('./wiki.txt', 'wb') as file_handler:
        file_handler.write(all_para.encode('utf8'))

url = 'https://sl2files.sustainablelivinglab.org/DatasetSocialMedia-Disaster-tweets-DFE.csv'
csv = ulib.request.urlopen(url).read()
with open('./socialmedia-disaster-tweets-DFE.csv', 'wb') as fx:
    fx.write(csv)

df_raw = p.read_csv('./socialmedia-disaster-tweets-DFE.csv', encoding='ISO-8859-1')

print ("You have successfully loaded your csv file")

print(df_raw.head(5))

print(list(df_raw['text'].sample()))

print(len(df_raw))

df_text = df_raw['text'].copy()
print(df_text)

frequecy_of_word=0
for i in range(len(df_raw)):
    x=df_raw["text"][i]
    if "stupid" in x:
        frequecy_of_word+=1
    else:
        continue

print("The frequency of the word stupid is :",frequecy_of_word)
    
