#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import time
import os
import sys
import collections
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms

from_numpy = torch.from_numpy

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import pickle
import gzip

#set path and load mnist data
os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/ift6285-tp1")
# os.chdir("/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework1/programming part ")
print(os.getcwd())


#%%
'''
# Load corpus data
'''
print('Load Corpus data:')
with gzip.open('data/train-1544.gz', 'rt', encoding='ISO-8859-1') as f:
    train0text = f.read()

print(len(train0text))

#%%
train0text=train0text.lower()
print(train0text[0:300])

# def corpus2list(corpus):
    
#%%
from textblob import TextBlob

text = '''The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
ngra1=blob.ngrams(n=2)
print(ngra1)
for ng in ngra1:
    print(ng)
#%%
# import nltk
# nltk.download()
# from nltk import bigrams
# bigram1=bigrams(text)
# for bg in bigram1:
#     print(bg)

#%%
import spacy
nlp = spacy.load('en')
doc = nlp(u'''The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
''')

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

#%%
print('----all token:')
for token in doc:
    print(token.text)

print('----lemma:')
for token in doc:
    print(token, token.lemma_, token.lemma)


print('----pos:')
for token in doc:
    print(token, token.pos_, token.pos)

print('----similarity:')
tokens = nlp(u'dog cat banana and chat')
for token1 in tokens:
    for token2 in tokens:
        print(token1.similarity(token2))

print('----命名实体识别:')
for ent in doc.ents:
    print(ent, ent.label_, ent.label)

print('-----vector')
print(doc[0])

print('------noun chunks:')
for nck in doc.noun_chunks:
    print(nck)

print('------sentances:')
for sent in doc.sents:
    print (sent)

#%%
import spacy
nlp = spacy.load('en')
s = ["thai iced tea",
"spicy fried chicken",
"sweet chili pork",
"thai chicken curry",]

def noun_notnoun(phrase):
    doc = nlp(phrase) # create spacy object
    token_not_noun = []
    notnoun_noun_list = []

    for item in doc:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            token_not_noun.append(item.text)
        if item.pos_ == "NOUN":
            noun = item.text

    for notnoun in token_not_noun:
        notnoun_noun_list.append(notnoun + " " + noun)

    return notnoun_noun_list

for phrase in s:
    print(noun_notnoun(phrase))


#%%
import textacy
import textacy.datasets
# text = '''I love you, and if you said, I don't care!'''
text='''The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.'''
doc = textacy.Doc(text)
print(type(doc))
print(doc)
# bag1=doc.to_bag_of_terms(ngrams=2, named_entities=True, lemmatize=True, as_strings=True)
bag1=doc.to_bag_of_terms(ngrams=2, normalize='lower',weighting='freq',as_strings=True,filter_stops=False)
print(type(bag1))
for k,v in bag1.items():
    print(k,v)
# bag2=doc.to_terms_list(ngrams=2, named_entities=True, lemmatize=True, as_strings=True)
# for k in bag2:
#     print(k)
#%%

tt=textacy.extract.ngrams(doc,2,filter_stops=False,filter_punct=True)
for t1 in tt:
    print(t1)


#%%
import textacy
#textacy中的token是包含标点符号的所有实例总数。
# tdoc = textacy.Doc('\t i am a cat \n, \t the cat is a dog \n, \t the dog is a student \n. \t a cat do like always a dog \n, \t the dog hate the mouse \n.', lang="en")
tdoc = textacy.Doc('bosbos i am a cat , the cat is a dogs , the dog is a student . a cat do like always a dog , the dog hate the mouse . eoseos', lang="en")
print(tdoc)
#如果normalize=lemma或空缺，那么会变成 i be, cat be
tdoc_bag2=tdoc.to_bag_of_terms(ngrams=2, normalize='lower',weighting='count',as_strings=True,filter_stops=False,filter_punct=False)
print('-----size of train lemm bag:',len(tdoc_bag2))
for k,v in tdoc_bag2.items():
    print('terms(', k,') ,Prob: ',v)

tdoc_bag1=tdoc.to_bag_of_terms(ngrams=1, normalize='lower',weighting='count',as_strings=True,filter_stops=False,filter_punct=False)
print('-----size of train lemm bag 1gram:',len(tdoc_bag1))
for k,v in tdoc_bag1.items():
    print('terms(', k,') ,Prob: ',v)

# probablite of n-gram:count(a cat)/(all tokens)
print('----(a cat) ',tdoc_bag2['a cat'])


print (tdoc.count('a cat'))
print('p(cat|a)=',tdoc_bag2['a cat']/tdoc.count('a'))
print('p(cat|a)=',tdoc_bag2['a cat']/tdoc_bag1['a'])
print('p(am|i)=',tdoc_bag['i am']/tdoc_bag1['i'])
print('p(the|bos)=',tdoc_bag['bosbos i']/tdoc_bag1['i'])


# tdoc_words=tdoc.to_bag_of_words(normalize='lower',weighting='count',as_strings=True)
# for k,v in tdoc_words.items():
#     print('words(', k,') ,Prob: ',v)

# tdoc_list=tdoc.to_terms_list(ngrams=2, normalize='lower',weighting='freq',as_strings=True,filter_stops=False)
# # print('size of train lemm bag:',len(tdoc_list))
# for k in tdoc_list:
#     print('list(', k,')')
#%%
print(len(tdoc.spacy_vocab))
for x in tdoc.spacy_vocab:
    if x=='reverse':
        print(type(x),x.text)

#%%

for sent in tdoc.sents:
    # if sent.text.startswith('i be'):
    # doc下的sent的类型是spacy.tokens.span.Span
    print(type(sent),sent)
    print(sent.vector)