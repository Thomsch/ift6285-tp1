#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# model2-hmm.py
# @author Zhibin.LU
# @created Fri Feb 23 2018 17:14:32 GMT-0500 (EST)
# @last-modified Mon Feb 26 2018 02:11:26 GMT-0500 (EST)
# @website: https://louis-udm.github.io
# @description 
# # # #


#%%

import time
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import gzip

import numpy as np
from hmmlearn import hmm
import spacy
import textacy
import regex as re

os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/ift6285-tp1")
print(os.getcwd())

'''
Load text in a string.
'''
def loadData2str(corpuspath):
    with gzip.open(corpuspath, 'rt', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    input_words=[]
    target_words=[]
    # for line in lines[: min(num_samples, len(lines) - 1)]:
    i=0
    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue
        line=line.encode("ascii", errors="ignore").decode()
        # if i<5:
        #     print(line)
        if len(line.split('\t'))==2:
            target_word, input_word = line.split('\t')
            input_word=input_word.lower().strip()
            target_word=target_word.lower().strip()
            if input_word.startswith("'") and not input_word.startswith("''"):
                input_word=input_word[1:]
            if target_word.startswith("'") and not target_word.startswith("''"):
                target_word=target_word[1:]
            if input_word=='':
                continue
            input_words.append(input_word)
            target_words.append(target_word)
            i+=1
    #         if i>=1 and i<=28:
    #             print(input_word,'|',target_word)
    # print('corpus tokens orignial: ',i)
    return ' '.join(input_words),' '.join(target_words)


train_lemm_corpus,train_surf_corpus=loadData2str('data/train-1544.gz')
test_lemm_corpus,test_surf_corpus=loadData2str('data/test-2834.gz')


#%%
'''
Get 2-gramms model, all types, all sentences of train_lemme set.
Get 2-gramms model, all types, all sentences of train_surface set.
'''
train_lemm_tacy_doc = textacy.Doc(train_lemm_corpus, lang="en")
print('train_lemm_tacy_doc: ',train_lemm_tacy_doc)
# bag1=doc.to_bag_of_terms(ngrams=2, named_entities=True, lemmatize=True, as_strings=True)
# bag_lemm=train_lemm_doc.to_bag_of_terms(ngrams=2, normalize='lower',weighting='freq',as_strings=True,filter_stops=False)
train_lemm_2grams_bag=train_lemm_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train lemm 2grams bag:',len(train_lemm_2grams_bag))
train_lemm_1grams_bag=train_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train lemm 1grams bag:',len(train_lemm_1grams_bag))

train_surf_tacy_doc = textacy.Doc(train_surf_corpus, lang="en")
print('train_surf_tacy_doc: ',train_surf_tacy_doc)
train_surf_2grams_bag=train_surf_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train surf 2grams bag:',len(train_surf_2grams_bag))
train_surf_1grams_bag=train_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train surf 1grams bag:',len(train_surf_1grams_bag))
train_lemm_tacy_sents=list(train_lemm_tacy_doc.sents)
train_surf_tacy_sents=list(train_surf_tacy_doc.sents)


'''
Get all types, all sentences of test_lemme set.
Get all types, all sentences of test_surface set.
'''
test_lemm_tacy_doc = textacy.Doc(test_lemm_corpus, lang="en")
print('test_lemm_tacy_doc: ',test_lemm_tacy_doc)
# test_lemm_2grams_bag=test_lemm_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
# print('size of test lemm 2grams bag:',len(test_lemm_2grams_bag))
test_lemm_1grams_bag=test_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of test lemm 1grams bag:',len(test_lemm_1grams_bag))

test_surf_tacy_doc = textacy.Doc(test_surf_corpus, lang="en")
print('test_surf_tacy_doc: ',test_surf_tacy_doc)
# test_surf_2grams_bag=test_surf_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
# print('size of test surf 2grams bag:',len(test_surf_2grams_bag))
test_surf_1grams_bag=test_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of test surf 1grams bag:',len(test_surf_1grams_bag))

test_lemm_tacy_sents=list(test_lemm_tacy_doc.sents)
test_surf_tacy_sents=list(test_surf_tacy_doc.sents)


#%%
# test
print(train_lemm_tacy_doc)
print(train_lemm_2grams_bag)
print(train_lemm_1grams_bag)
print('loss . ',train_lemm_2grams_bag['loss .'])
print('. the',train_lemm_2grams_bag['. the'])
i=0
for sent in train_lemm_tacy_doc.sents:
    print(sent.text)
    i+=1
    if i>10: break

# for i,chs in enumerate(zip(train_lemm_tacy_doc.tokens,train_surf_tacy_doc.tokens)):
#     # if chs[0].text=='have' and chs[1].text=="'":
#     #     print(i,chs[0],chs[1])
#     #     break
#     if chs[0].text not in ['be','find','get','have','a','he','lie','use','leave','go','see','she','we','i','would'] and chs[0].text[0]!=chs[1].text[0]:
#         print(i,chs[0],chs[1])
#         break
#     # if i>=740 and i<=750:
#     #     print(i,chs[0],chs[1])

# #%%
# # print(train_lemm_corpus[0:200])
# for i,chs in enumerate(zip(train_lemm_tacy_doc.tokens,train_lemm_corpus.split(' '))):
#     if chs[0].text!=chs[1]:
#         print(i,'|'+chs[0].text+'|','|'+chs[1]+'|')
#         # break
#     if i>345:
#         break

#%%
'''
Get all pair of surf-lemma and their count on train data set.
'''
pairs_list=[]
for lemma,surf in zip(train_lemm_tacy_doc, train_surf_tacy_doc):
    pairs_list.append(surf.text+' '+lemma.text)
# print(pairs_list[0],pairs_list.count(pairs_list[0]))
# pairs_list=np.array(pairs_list)
train_surf_lemm_map={}
for i,pair in enumerate(pairs_list):
    if pair not in train_surf_lemm_map:
        train_surf_lemm_map[pair]=pairs_list.count(pair)

#test
print('are be ',train_surf_lemm_map['are be'])
print('( ( ',train_surf_lemm_map['( ('])
print('. . ',train_surf_lemm_map['. .'])

#%%
#test
print('(rimatara reed) ',train_lemm_2grams_bag['rimatara reed'])
print('(you be) ',train_lemm_2grams_bag['you be'])
print('(he go) ',train_lemm_2grams_bag['he go'])

print('p(be|you)=',train_lemm_2grams_bag['you be']/train_lemm_1grams_bag['you'])
print('p(cat|a)=',train_lemm_2grams_bag['a cat']/train_lemm_1grams_bag['a'])
print('p(am|i)=',train_surf_2grams_bag['i am']/train_surf_1grams_bag['i'])
print('p(be-o|are-s)=',train_surf_lemm_map['are be']/train_surf_1grams_bag['are'])
print('p(.-o|.-s)=',train_surf_lemm_map['. .']/train_surf_1grams_bag['.'])
# print('p(the|bos)=',train_surf_2grams_bag['. the'])
# print(train_surf_1grams_bag)

# i=0
# for k,v in train_lemm_2grams_bag.items():
#     print('(', k,') ,Prob: ',v)
#     i+=1
#     if i>50: break
# bag2=doc.to_terms_list(ngrams=2, named_entities=True, lemmatize=True, as_strings=True)
# for k in bag2:
#     print(k)


#%%
'''
Prepare all data of model HMM:
'''
states=sorted(list( train_surf_1grams_bag.keys() | test_surf_1grams_bag.keys() ))
states_map=dict(
    [(typex, i) for i, typex in enumerate(states)])
n_states=len(states)

observations=sorted(list( train_lemm_1grams_bag.keys() | test_lemm_1grams_bag.keys() ))
observations_map=dict(
    [(typey, i) for i, typey in enumerate(observations)])
n_observations=len(observations)

#%%
start_probability=np.zeros(n_states)
n_sents=len(list(train_surf_tacy_doc.sents))
for sent in train_surf_tacy_doc.sents:
    head=sent[0].text
    if head not in states_map:
        head=sent[1].text
    if head not in states_map:
        head=sent[2].text
    if head not in states_map:
        head=sent[3].text
    if head not in states_map:
        head=sent[4].text
    if head not in states_map:
        head=sent[5].text
    start_probability[states_map[head]]+=1

start_probability=start_probability/n_sents

#test
print ('start_probability: ',start_probability[0:5])

#%%
# print('p(am|i)=',train_surf_2grams_bag['i am']/train_surf_1grams_bag['i'])

transition_probability=np.zeros((n_states,n_states))
for k,v in train_surf_2grams_bag.items():
    if len(k.split(' '))<2:
        # print(k)
        continue
    type_prev,type_curr=k.split(' ')
    # if  transition_probabilitynp[states_map[type_prev],states_map[type_curr]]>0:
    #     continue
    prob=train_surf_2grams_bag[k]/train_surf_1grams_bag[type_prev]
    # print('p(',type_curr,'|',type_prev,')=',prob)
    transition_probability[states_map[type_prev],states_map[type_curr]]=prob

# The sum of the probability values for some lines <1, I don't know why
# and the missing probability needs to be filled.
residu=((1-transition_probability.sum(1))/n_states)[:,None]
transition_probability=transition_probability+residu

#%%
# print('p(be-o|are-s)=',train_surf_lemm_map['are be']/train_surf_1grams_bag['are'])

emission_probability=np.zeros((n_states,n_observations))
for k,v in train_surf_lemm_map.items():
    if len(k.split(' '))<2:
        # print(k)
        continue
    type_s,type_o=k.split(' ')
    if type_s not in train_surf_1grams_bag:
        continue
    prob=train_surf_lemm_map[k]/train_surf_1grams_bag[type_s]
    emission_probability[states_map[type_s],observations_map[type_o]]=prob
# print ('emission_probability: ',emission_probability[0:5,0])

#%%
'''
Transforme all these sentences to vector.
'''
dot_lemm_index=observations_map['.']
dot_surf_index=states_map['.']
max_lemm_sent=max([max([len(sent) for sent in train_lemm_tacy_sents]),max([len(sent) for sent in test_lemm_tacy_sents])])
max_surf_sent=max([max([len(sent) for sent in train_surf_tacy_sents]),max([len(sent) for sent in test_surf_tacy_sents])])

train_lemm_vectors=np.zeros((len(train_lemm_tacy_sents),max_lemm_sent),dtype=np.int16)
train_lemm_vectors+=dot_lemm_index
for i,sent in enumerate(train_lemm_tacy_sents):
    for j,lemm in enumerate(sent):
        train_lemm_vectors[i,j]=observations_map[lemm.text]

# TODO: Must deal with those type who appear in the test, but not in the train.
test_lemm_vectors=np.zeros((len(test_lemm_tacy_sents),max_lemm_sent),dtype=np.int16)
test_lemm_vectors+=dot_lemm_index
for i,sent in enumerate(test_lemm_tacy_sents):
    for j,lemm in enumerate(sent):
        if lemm.text in observations_map:
            test_lemm_vectors[i,j]=observations_map[lemm.text]
        # TODO:Must deal with those who appear in the test, but not in the train type
        # else:
        #     sample[i]=

train_surf_vectors=np.zeros((len(train_surf_tacy_sents),max_surf_sent),dtype=np.int16)
train_surf_vectors+=dot_surf_index
for i,sent in enumerate(train_surf_tacy_sents):
    for j,surf in enumerate(sent):
        train_surf_vectors[i,j]=states_map[surf.text]

# TODO: Must deal with those type who appear in the test, but not in the train.
test_surf_vectors=np.zeros((len(test_surf_tacy_sents),max_surf_sent),dtype=np.int16)
test_surf_vectors+=dot_surf_index
for i,sent in enumerate(test_surf_tacy_sents):
    for j,surf in enumerate(sent):
        if surf.text in states_map:
            test_surf_vectors[i,j]=states_map[surf.text]
        # TODO:Must deal with those who appear in the test, but not in the train type
        # else:
        #     sample[i]=

#%%
print('train_hmm_vectors, len: ',len(train_lemm_vectors),'[1]:',train_lemm_vectors[1])
print(train_lemm_tacy_sents[1])
print(train_surf_tacy_sents[1])
print('test_hmm_vectors, len: ',len(test_lemm_vectors),'[2]:',test_lemm_vectors[2])
print(test_lemm_tacy_sents[2])
print(test_surf_tacy_sents[2])

#%%
'''
Define the HMM Model
'''
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_= start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

'''
Evalutate the prediction
'''
def accuracy(vectors_pred,vectors_origin):
    compare=vectors_pred==vectors_origin
    compare.reshape(-1)
    return compare.count(False)/len(compare)

def decode_sents(vectors,type_list):
    sents=[]
    for v in vectors:
        sent=' '.join(map(lambda x: type_list[x], v))
        # print (sent)
        sents.append(sent)
    return sents

#%%
'''
Prediction
'''
lemm_seqs=train_lemm_vectors[0:3]
target_seqs=train_surf_vectors[0:3]
target_surf_origin=train_surf_tacy_sents[0:3]
# X : array-like, shape (n_samples, n_features)
# logprob, output_seq = model.decode(input_seq.reshape(-1, 1), algorithm="viterbi")
logprob, predict_seqs = model.decode(lemm_seqs, algorithm="viterbi")
#%%
lemm_seq2sents=decode_sents(lemm_seqs,observations)
pred_seq2sents=decode_sents(predict_seqs,states)
print('predict on training data, result:')
for s1,s2,s3 in zip(lemm_seq2sents,target_surf_origin,pred_seq2sents):
    print('--')
    print(s1)
    print(s2)
    print(s3)

print('Accuracy: ', accuracy(predict_seqs,target_seqs))

# print (logprob)
#%%

lemm_seqs=test_lemm_vectors[0:3]
target_seqs=test_surf_vectors[0:3]
target_surf_origin=test_surf_tacy_sents[0:3]
# X : array-like, shape (n_samples, n_features)
# logprob, output_seq = model.decode(input_seq.reshape(-1, 1), algorithm="viterbi")
logprob, predict_seqs = model.decode(lemm_seqs, algorithm="viterbi")
#%%
lemm_seq2sents=decode_sents(lemm_seqs,observations)
pred_seq2sents=decode_sents(predict_seqs,states)
print('predict on test data, result:')
for s1,s2,s3 in zip(lemm_seq2sents,target_surf_origin,pred_seq2sents):
    print('--')
    print(s1)
    print(s2)
    print(s3)

print('Accuracy: ', accuracy(predict_seqs,target_seqs))


