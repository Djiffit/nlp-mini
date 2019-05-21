from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer = WordNetLemmatizer()

def process_text(text):
    sents = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sents]
    pos = [pos_tag(sent) for sent in words]
    lem = [[lemmatizer.lemmatize(word) for word in sent] for sent in words]
    
    assert(len(words) == len(pos) and len(words) == len(lem))
    assert(len(words[0]) == len(pos[0]) and len(words[0]) == len(lem[0]))
    
    return (words, pos, lem)
    
text = 'Build your own NLP pipeline (a function named process_text(text)) that takes a paragraph as input, and splits the paragraph into sentences, applies word tokenization, POS tagging and lemmatization on all words. The function should return a list containing the processed sentences.'

process = process_text(text)

print(process)


import re

stopWords = set(stopwords.words('english'))
accepted_pos = re.compile('(NN.?.?|VB.?|JJ.?)')

def filter_text(text):
    words, pos, lem = process_text(text)
    filtered = []
    
    for sind, sent in enumerate(lem):
        for wind, word in enumerate(sent):
            if word not in stopWords and accepted_pos.match(pos[sind][wind][1]):
                filtered += [words[sind][wind]]
                
    return filtered

text = 'Implement a function (filter_text(text)) that uses process_text(text) to process a paragraph and then removes stop words and words that are not verbs, adjectives or nouns (for descriptions of POS tags, read this). Here is a second sentence to wow everyone how well this thing is truly working. What a masterpiece of software.' 
print(filter_text(text))
                
import spacy 
nlp = spacy.load("en") 
    
def spacy_parser(text):
    doc = nlp(text)
    for sent in doc.sents:
        for token in sent:
            print(token.text, token.pos_, token.tag_, token.dep_)
            
def parse_compare(text):
    print(spacy_parser(text))
    print(process_text(text))

parse_compare('I have a dog')

parse_compare('Finger licking good.')

parse_compare('Finger Lickin\' Good')

parse_compare('Think different.')

import requests, xmltodict, pickle, os
import numpy as np

def totally_pun_word(word):
    res = requests.get(f'https://api.datamuse.com/words?sl={word}').json()
    rand = np.random.randint(0, len(res))
    return res[rand]['word']

def make_punny(text):
    fil = filter_text(text)
    rand = np.random.randint(0, len(fil))
    replace = fil[rand]
    replacer = totally_pun_word(fil[rand])
    text = text.replace(replace, replacer)
    
    return text
    
print(make_punny('Jurassic park'))
print(make_punny('Jurassic park'))
print(make_punny('Jurassic park'))
print(make_punny('Jurassic park'))
print(make_punny('Life of Pi'))
print(make_punny('Life of Pi'))
print(make_punny('Life of Pi'))
print(make_punny('Life of Pi'))
print(make_punny('Life of Pi'))
print(make_punny('Game of Thrones'))
print(make_punny('Game of Thrones'))
print(make_punny('Game of Thrones'))
print(make_punny('Game of Thrones'))
print(make_punny('Lord of the Rings'))
print(make_punny('Lord of the Rings'))
print(make_punny('Lord of the Rings'))
print(make_punny('Lord of the Rings'))