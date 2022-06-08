#!pip install numpy scipy
#!pip install scikit-learn
#!pip install pillow
#!pip install h5py

#!pip install tensorflow
#!pip install tensorflow-gpu
#!pip install keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
 
from numpy import argmax
import numpy as np
import re
import requests as rq
import bs4 as b
import urllib as ulib
import pandas as p
import nltk # to process text data
import numpy as np # to represent corpus as arrays
import random 
import string # to process standard python strings
from sklearn.metrics.pairwise import cosine_similarity # We will use this later to decide how similar two sentences are
from sklearn.feature_extraction.text import TfidfVectorizer # Remember when you built a function to create a tfidf bag of words in Experience 2? This function does the same thing!

nltk.download('punkt') # first-time use only tokenizer
nltk.download('wordnet') # first-time use only Used for the lemmatizer

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

'''
When you have collected your data, update the variable 'filepath' below with the location of your knowledge base. 
The knowledge base should consist of sentences in a text file.
'''
filepath='./robots.txt'
corpus=open(filepath,'r',errors = 'ignore')
raw_data=corpus.read()
print (raw_data)

raw_data=raw_data.lower()# converts to lowercase
print (raw_data)

lemmer = nltk.stem.WordNetLemmatizer() #Initiate lemmer class. WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) #see previous section 1.2.5 lemmatization

test_sentence='Today was a wonderful day. The sun was shining so brightly and the birds were chirping loudly!'
test_word_tokens = nltk.word_tokenize()

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

print(LemTokens())

GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split(): # Looks at each word in your sentence
        if word.lower() in GREETING_INPUTS: # checks if the word matches a GREETING_INPUT
            return random.choice(GREETING_RESPONSES) # replies with a GREETING_RESPONSE

def response(user_response):
    
    robo_response='' # initialize a variable to contain string
    sent_tokens.append(user_response) #add user response to sent_tokens
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens) #get tfidf value
    vals = cosine_similarity(tfidf[-1], tfidf) #get cosine similarity value
    idx=vals.argsort()[0][-2] 
    flat = vals.flatten() 
    flat.sort() #sort in ascending order
    req_tfidf = flat[-2] 
    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

import datetime

def tell_time(sentence):
    for word in sentence.split():
        # your code here
            currentdt = datetime.datetime.now()
            return currentdt.strftime("%Y-%m-%d %H:%M:%S")

tell_time('time')

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
                # Uncomment the statement below once you have written your tell_time fuction.
#             if(tell_time(user_response)!=None):
#                 print("ROBO: "+tell_time(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
}
call = requests.get('https://api.openrouteservice.org/v2/directions/driving-car?api_key=5b3ce3597851110001cf6248818bd962dc594b54a9b3115672f57e92&start=8.681495,49.41461&end=8.687872,49.420318', headers=headers)

print(call.status_code, call.reason)
print(call.text)

X = ['Hi',
     'Hello',
     'How are you?',
     'I am making',
     'making',
     'working',
     'studying',
     'see you later',
     'bye',
     'goodbye']

print(len(X))

Y = ['greeting',
     'greeting',
     'greeting',
     'busy',
     'busy',
     'busy',
     'busy',
     'bye',
     'bye',
     'bye']

print(len(Y))

def remove_non_alpha_numeric_characters(sentence):
    new_sentence = ''
    for alphabet in sentence:
        if alphabet.isalpha() or alphabet == ' ':
            new_sentence += alphabet
    return new_sentence

def preprocess_data(X):
    X = [data_point.lower() for data_point in X]
    X = [remove_non_alpha_numeric_characters(
        sentence) for sentence in X]
    X = [data_point.strip() for data_point in X]
    X = [re.sub(' +', ' ',
                data_point) for data_point in X]
    return X

X = preprocess_data(X)

vocabulary = set()
for data_point in X:
    for word in data_point.split(' '):
        vocabulary.add(word)

vocabulary = list(vocabulary)

X_encoded = []

def encode_sentence(sentence):
    sentence = preprocess_data([sentence])[0]
    sentence_encoded = [0] * len(vocabulary)
    for i in range(len(vocabulary)):
        if vocabulary[i] in sentence.split(' '):
            sentence_encoded[i] = 1
    return sentence_encoded

X_encoded = [encode_sentence(sentence) for sentence in X]

classes = list(set(Y))

Y_encoded = []
for data_point in Y:
    data_point_encoded = [0] * len(classes)
    for i in range(len(classes)):
        if classes[i] == data_point:
            data_point_encoded[i] = 1
    Y_encoded.append(data_point_encoded)

X_train = X_encoded
y_train = Y_encoded
X_test = X_encoded
y_test = Y_encoded

print (y_test)

print(len(X_train))

model = Sequential()
model.add(Dense(units=64, activation='sigmoid',
                input_dim=len(X_train[0])))
model.add(Dense(units=len(y_train[0]), activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.01,
                            momentum=0.9, nesterov=True))
model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=16)

predictions = [argmax(pred) for pred in model.predict(np.array(X_test))]

correct = 0
for i in range(len(predictions)):
    if predictions[i] == argmax(y_test[i]):
        correct += 1

print ("Correct:", correct)
print ("Total:", len(predictions))


while True:
    print ("Enter a sentence")
    sentence = input()
    prediction= model.predict(np.array([encode_sentence(sentence)]))
    print (classes[argmax(prediction)])

