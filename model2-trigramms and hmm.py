#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# model2-hmm.py
# @author Zhibin.LU
# @created Fri Feb 23 2018 17:14:32 GMT-0500 (EST)
# @last-modified Wed Mar 07 2018 15:27:23 GMT-0500 (EST)
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
from collections import Counter

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/TP1/ift6285-tp1")
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
            # if input_word.startswith("'") and not input_word.startswith("''"):
            #     input_word=input_word[1:]
            # if target_word.startswith("'") and not target_word.startswith("''"):
            #     target_word=target_word[1:]
            # pattern = re.compile(r'[-+]')
            # input_word=re.sub(pattern, ' ', input_word)
            # target_word=re.sub(pattern, ' ', target_word)
            pattern = re.compile(r'\'')
            input_word=re.sub(pattern, '', input_word)
            target_word=re.sub(pattern, '', target_word)

            input_word=re.sub('([-+/\.])', r" \1 ",input_word)
            target_word=re.sub('([-+/\.])', r" \1 ",target_word)

            pattern = re.compile(r'\d+s')
            m1=re.search(pattern, input_word)
            m2=re.search(pattern, target_word)
            if m2 is not None and m1 is None:
                input_word=re.sub('(\d+)', r"\1s", input_word)

            input_word=re.sub('(\d+)', r" \1 ", input_word)
            target_word=re.sub('(\d+)', r" \1 ", target_word)

            input_word=re.sub(' +', ' ', input_word)
            target_word=re.sub(' +', ' ', target_word)
            if input_word=='':
                continue
            input_words.append(input_word)
            target_words.append(target_word)
            i+=1
            # if i>235230 and i<235280:
            #     print(input_word,target_word)
            #     print('++++++',i)
    #         if i>=1 and i<=28:
    #             print(input_word,'|',target_word)
    # print('corpus tokens orignial: ',i)
    return ' '.join(input_words),' '.join(target_words)

train_lemm_corpus,train_surf_corpus=loadData2str('data/train-1183.gz')
test_lemm_corpus,test_surf_corpus=loadData2str('data/test-2834.gz')

train_lemm_corpus=re.sub(' +', ' ', train_lemm_corpus)
train_surf_corpus=re.sub(' +', ' ', train_surf_corpus)
test_lemm_corpus=re.sub(' +', ' ', test_lemm_corpus)
test_surf_corpus=re.sub(' +', ' ', test_surf_corpus)

#%%
'''
Get 2-gramms model, all types, all sentences of train_lemme set.
Get 2-gramms model, all types, all sentences of train_surface set.
Get all types, all sentences of test_lemme set.
Get all types, all sentences of test_surface set.
'''
# train_lemm_tacy_doc = textacy.Doc(train_lemm_corpus, lang="en")
# train_surf_tacy_doc = textacy.Doc(train_surf_corpus, lang="en")

nlp = English()
# nlp = spacy.load('en', disable=['parser', 'tagger'])
train_lemm_tacy_doc=nlp(train_lemm_corpus)
train_surf_tacy_doc=nlp(train_surf_corpus)
test_lemm_tacy_doc =nlp(test_lemm_corpus)
test_surf_tacy_doc =nlp(test_surf_corpus)

print('Tokens of train_lemm_tacy_doc: ',len(train_lemm_tacy_doc))
print('Tokens of train_surf_tacy_doc: ',len(train_surf_tacy_doc))
if len(train_lemm_tacy_doc)!=len(train_surf_tacy_doc):
    print('Warning: the numbre of tokens of lemme and surfaceis in train not equal !!!!!!')

print('Tokens of test_lemm_tacy_doc: ',len(test_lemm_tacy_doc))
print('Tokens of test_surf_tacy_doc: ',len(test_surf_tacy_doc))
if len(test_lemm_tacy_doc)!=len(test_surf_tacy_doc):
    print('Warning: the numbre of tokens of lemme and surfaceis on test not equal !!!!!!')

# #%%
# #test
# for i in range(30):
#     print(doc1[1600+i])
#     print(doc2[1600+i])
#     print('-----')

train_surf_tacy_sents=[]
start_ind=0
for token in train_surf_tacy_doc:
    if token.text in ['.','?','!']:
        train_surf_tacy_sents.append(train_surf_tacy_doc[start_ind:token.i+1])
        start_ind=token.i+1
print('total sentence of train surf:',len(train_surf_tacy_sents))
train_lemm_tacy_sents=[]
start_ind=0
for token in train_lemm_tacy_doc:
    if token.text in ['.','?','!']:
        train_lemm_tacy_sents.append(train_lemm_tacy_doc[start_ind:token.i+1])
        start_ind=token.i+1
print('total sentence of train lemm:',len(train_lemm_tacy_sents))

if len(train_surf_tacy_sents)!=len(train_lemm_tacy_sents):
    print('Warning: the numbre of sentances of lemme and surface is not equal !!!!!!')

test_surf_tacy_sents=[]
start_ind=0
for token in test_surf_tacy_doc:
    if token.text in ['.','?','!']:
        test_surf_tacy_sents.append(test_surf_tacy_doc[start_ind:token.i+1])
        start_ind=token.i+1
print('total sentence of test surf:',len(test_surf_tacy_sents))
test_lemm_tacy_sents=[]
start_ind=0
for token in test_lemm_tacy_doc:
    if token.text in ['.','?','!']:
        test_lemm_tacy_sents.append(test_lemm_tacy_doc[start_ind:token.i+1])
        start_ind=token.i+1
print('total sentence of test lemm:',len(test_lemm_tacy_sents))

if len(test_surf_tacy_sents)!=len(test_lemm_tacy_sents):
    print('Warning: the numbre of sentances of lemme and surface on test is not equal !!!!!!')

#%%
train_lemm_tacy_doc = textacy.Doc(train_lemm_tacy_doc)
train_surf_tacy_doc = textacy.Doc(train_surf_tacy_doc)

# bag1=doc.to_bag_of_terms(ngrams=2, named_entities=True, lemmatize=True, as_strings=True)
# bag_lemm=train_lemm_doc.to_bag_of_terms(ngrams=2, normalize='lower',weighting='freq',as_strings=True,filter_stops=False)
train_lemm_2grams_bag=train_lemm_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train lemm 2grams bag:',len(train_lemm_2grams_bag))
train_lemm_1grams_bag=train_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train lemm 1grams bag:',len(train_lemm_1grams_bag))

train_surf_2grams_bag=train_surf_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train surf 2grams bag:',len(train_surf_2grams_bag))
train_surf_1grams_bag=train_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of train surf 1grams bag:',len(train_surf_1grams_bag))


test_lemm_tacy_doc = textacy.Doc(test_lemm_tacy_doc)
test_surf_tacy_doc = textacy.Doc(test_surf_tacy_doc)

# test_lemm_2grams_bag=test_lemm_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
# print('size of test lemm 2grams bag:',len(test_lemm_2grams_bag))
test_lemm_1grams_bag=test_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of test lemm 1grams bag:',len(test_lemm_1grams_bag))

# test_surf_2grams_bag=test_surf_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
# print('size of test surf 2grams bag:',len(test_surf_2grams_bag))
test_surf_1grams_bag=test_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower',named_entities=False, weighting='count',as_strings=True,filter_stops=False,filter_punct=False,filter_nums=False,drop_determiners=False)
print('size of test surf 1grams bag:',len(test_surf_1grams_bag))

#%%
# test
print(type(train_lemm_2grams_bag),len(train_lemm_2grams_bag))
print(type(train_lemm_1grams_bag),len(train_lemm_2grams_bag))
print('him . ',train_lemm_2grams_bag['him .'])
print('. the',train_lemm_2grams_bag['. the'])
i=0
for sent in train_lemm_tacy_sents:
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
    pairs_list.append(surf.text.strip()+' '+lemma.text.strip())
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
# print('(rimatara reed) ',train_lemm_2grams_bag['rimatara reed'])
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
Functions of Evalutate the prediction
'''

'''
# test accuracy, Raw accuracy
'''
def count_accuracy_raw(pred_corpus,target_corpus):
    count_accu=0
    total=0
    pred_sents=pred_corpus.split('.')
    target_sents=target_corpus.split('.')
    for pred_sent,target_sent in zip(pred_sents,target_sents):
        pred_list=pred_sent.split(' ')
        targ_list=target_sent.split(' ')
        for pred_token,target_token in zip(pred_list,targ_list):
            total+=1
            if pred_token==target_token:
                count_accu+=1
    return count_accu, total

raw_acc_count,raw_count_total=count_accuracy_raw(train_lemm_corpus,train_surf_corpus)
print('test of Accuracy raw:', raw_acc_count,'/', raw_count_total,'=',raw_acc_count/raw_count_total)

'''
# test accuracy,  accuracy of spacy's token
'''
def count_accuracy_spacy(pred_sents,target_sents):
    count_accu=0
    total=0
    for pred_sent,target_sent in zip(pred_sents,target_sents):
        total+=1
        # if pred_sent.text[0] != target_sent.text[0] and target_sent.text \
        #     not in ['be','she','go','we','i']:
        #     print(preword,pred_sent)
        #     print(pretarg,target_sent)
        #     print('-----',total)
        # preword=pred_sent
        # pretarg=target_sent
        for pred_token,target_token in zip(pred_sent,target_sent):
            total+=1
            if pred_token.text==target_token.text:
                count_accu+=1
            # if pred_token.text=='.' and target_token.text!='.':
            # if total>239980 and total<240060:
            # if total==240047:
            #     print(pred_sent)
            #     print(target_sent)
    return count_accu, total

spacy_acc_count,spacy_count_total=count_accuracy_spacy(train_lemm_tacy_sents,train_surf_tacy_sents)
print('test of Accuracy spacy:', spacy_acc_count,'/', spacy_count_total,'=',spacy_acc_count/spacy_count_total)

def count_accuracy(pred_sents,target_sents):
    count_accu=0
    total=0
    for pred_sent,target_sent in zip(pred_sents,target_sents):
        pred_list=re.split(r"-| |\?",pred_sent)
        # pred_list=pred_sent.split(' ')
        for pred_token,target_token in zip(pred_list,target_sent):
            total+=1
            if pred_token==target_token.text:
                count_accu+=1
    return count_accu, total

def decode_sents(vectors,type_list):
    sents=[]
    for v in vectors:
        sent=' '.join(map(lambda x: type_list[x], v))
        # print (sent)
        sents.append(sent)
    return sents

def decode_sent(vector,type_list):
    return ' '.join(map(lambda x: type_list[x], vector))




#%%
'''
**** Model Bi-gramms predicteur ****
'''
'''
get all  [lemm(t-1),lemm(t)] -> surf(t) 
and get map of bi-gramms [lemm(t-1),lemm(t)] -> surf word , 
in which the surface word is max count of the same pair of [lemm(t-1),lemm(t)].
for example: if there have {[you be]->are} 3 times, and {[you be]->is} 1 times,
then map([you be])=are.
'''
bigramms_lemm_surf_map={}
bigramms_lemm_surf_count_map={}
for lemm_sent,surf_sent in zip(train_lemm_tacy_sents,train_surf_tacy_sents):
    for i,token in enumerate( zip(lemm_sent, surf_sent)):
        if i==0:
            # bigramms_lemm_surf_map[token[0].text]=token[1].text
            if token[0].text in bigramms_lemm_surf_count_map:
                l1=bigramms_lemm_surf_count_map[token[0].text]
                l1.append(token[1].text)
                # bigramms_lemm_surf_count_map[token[0].text]=l1
            else:
                bigramms_lemm_surf_count_map[token[0].text]=[token[1].text]
            lemm_pre=token[0].text
        else:
            # if token[0].text=='be' and lemm_pre=='you':print(token[1].text)
            # bigramms_lemm_surf_map[lemm_pre+' '+token[0].text]=token[1].text
            if lemm_pre+' '+token[0].text in bigramms_lemm_surf_count_map:
                l1=bigramms_lemm_surf_count_map[lemm_pre+' '+token[0].text]
                l1.append(token[1].text)
                # bigramms_lemm_surf_count_map[lemm_pre+' '+token[0].text]=l1
            else:
                bigramms_lemm_surf_count_map[lemm_pre+' '+token[0].text]=[token[1].text]
            lemm_pre=token[0].text

for k,v in bigramms_lemm_surf_count_map.items():
    word_counts = Counter(v)
    bigramms_lemm_surf_map[k]=word_counts.most_common(1)[0][0]

print('size of bi-grammes: ',len(bigramms_lemm_surf_map))
#test
print('you be -> ',bigramms_lemm_surf_map['you be'])

#%%
print('--Model Bi-gramms predicteur predict on test data:---')
bigramms_pred_sents=[]
count_accu=0
for k,sent in enumerate( zip(test_lemm_tacy_sents,test_surf_tacy_sents)):
    pred_sent=[]
    for i,token in enumerate(zip(sent[0],sent[1])):
        if i==0:
            if token[0].text in bigramms_lemm_surf_map:
                pred_token=bigramms_lemm_surf_map[token[0].text]
                if pred_token==token[1].text:
                    count_accu+=1
                pred_sent.append(pred_token)
            else:
                # if can't find the pair of this lemm word,use directly this lemm word
                pred_sent.append(token[0].text)
                # if this not paired lemm word ==the surface word correspondant.
                if token[0].text==token[1].text:
                    count_accu+=1
            lemm_pre=token[0].text
        else:
            if lemm_pre+' '+token[0].text in bigramms_lemm_surf_map:
                pred_token=bigramms_lemm_surf_map[lemm_pre+' '+token[0].text]
                if pred_token==token[1].text:
                    count_accu+=1
                pred_sent.append(pred_token)
            else:
                # if can't find the pair of this lemm word,use directly this lemm word
                pred_sent.append(token[0].text)
                # if this not paired lemm word ==the surface word correspondant.
                if token[0].text==token[1].text:
                    count_accu+=1
            lemm_pre=token[0].text

    pred_sent_text=' '.join(pred_sent)
    # pred_sent_text=pred_sent_text.rstrip()
    bigramms_pred_sents.append(pred_sent_text)
    if k<=30:
        print('-- NO.',k)
        print(test_lemm_tacy_sents[k].text)
        print(test_surf_tacy_sents[k].text)
        print(pred_sent_text)

print('Accuracy of bi-gramms predicteur on test data:', count_accu,'/', len(test_surf_tacy_doc),'=',count_accu/len(test_surf_tacy_doc))

raw_acc_count,raw_count_total=count_accuracy_raw(test_lemm_corpus,test_surf_corpus)
print('Accuracy raw on test data:', raw_acc_count,'/', raw_count_total,'=',raw_acc_count/raw_count_total)


#%%
'''
**** Model HMM predicteur with given all matrix of probability ****
'''

'''
Prepare all parameters of model HMM:
'''
# states=sorted(list( train_surf_1grams_bag.keys() | test_surf_1grams_bag.keys() ))
states=sorted(list( train_surf_1grams_bag.keys() ))
states_map=dict(
    [(typex, i) for i, typex in enumerate(states)])
n_states=len(states)

# observations=sorted(list( train_lemm_1grams_bag.keys() | test_lemm_1grams_bag.keys() ))
observations=sorted(list( train_lemm_1grams_bag.keys() ))
observations_map=dict(
    [(typey, i) for i, typey in enumerate(observations)])
n_observations=len(observations)

#%%
start_probability=np.zeros(n_states)
for sent in train_surf_tacy_sents:
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

start_probability=start_probability/len(train_surf_tacy_sents)

#test
print ('start_probability: ',start_probability[0:5])

#%%
# get the count of all first type in bi-gramms:
train_surf_first_type_bag={}
for k,v in train_surf_2grams_bag.items():
    # if k.startswith('_'): print(k)
    if len(k.split(' '))!=2:
        # print(k)
        continue
    type_prev,type_curr=k.split(' ')
    if type_prev not in train_surf_first_type_bag:
        train_surf_first_type_bag[type_prev]=v
    else:
        train_surf_first_type_bag[type_prev]+=v

# print('p(am|i)=',train_surf_2grams_bag['i am']/train_surf_first_type_bag['i'])
transition_probability=np.zeros((n_states,n_states))
for k,v in train_surf_2grams_bag.items():
    if len(k.split(' '))!=2:
        # print(k)
        continue
    type_prev,type_curr=k.split(' ')
    # if  transition_probabilitynp[states_map[type_prev],states_map[type_curr]]>0:
    #     continue
    prob=train_surf_2grams_bag[k]/train_surf_first_type_bag[type_prev]
    # print('p(',type_curr,'|',type_prev,')=',prob)
    transition_probability[states_map[type_prev],states_map[type_curr]]=prob
    # if states_map[type_prev]==3: 
    #     print(type_prev,type_curr,prob)
    #     print(k,v)
    #     print(train_surf_first_type_bag[type_prev])n_states

# smooth probablity:some type didn't apeare in train_surf_2grams_bag
for i,prob in enumerate(transition_probability.sum(1)):
    if prob==0: 
        transition_probability[i]+=1.0/n_states
# residu=((1-transition_probability.sum(1))/n_states)[:,None]
# transition_probability=transition_probability+residu

#%%
# print('p(be-observ|are-stat)=',train_surf_lemm_map['are be']/train_surf_1grams_bag['are'])

emission_probability=np.zeros((n_states,n_observations))
for k,v in train_surf_lemm_map.items():
    # k=k.strip()
    if len(k.split(' '))!=2:
        print('surf-lemm group error:',k)
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
print('max_lemm_sent:',max_lemm_sent)
print('max_surf_sent:',max_surf_sent)

train_lemm_vectors=[]
for i,sent in enumerate(train_lemm_tacy_sents):
    sample=np.zeros(len(sent),dtype=np.int16)
    sample+=dot_lemm_index
    for j,lemm in enumerate(sent):
        sample[j]=observations_map[lemm.text]
    train_lemm_vectors.append(sample)

# TODO: Must deal with those type who appear in the test, but not in the train.
test_lemm_vectors=[]
for i,sent in enumerate(test_lemm_tacy_sents):
    sample=np.zeros(len(sent),dtype=np.int16)    
    sample+=dot_lemm_index
    for j,lemm in enumerate(sent):
        if lemm.text in observations_map:
            sample[j]=observations_map[lemm.text]
    test_lemm_vectors.append(sample)
        # TODO:Must deal with those who appear in the test, but not in the train type
        # else:
        #     sample[i]=

# train_surf_vectors=np.zeros((len(train_surf_tacy_sents),max_surf_sent),dtype=np.int16)
# train_surf_vectors+=dot_surf_index
# for i,sent in enumerate(train_surf_tacy_sents):
#     for j,surf in enumerate(sent):
#         train_surf_vectors[i,j]=states_map[surf.text]

# # TODO: Must deal with those type who appear in the test, but not in the train.
# test_surf_vectors=np.zeros((len(test_surf_tacy_sents),max_surf_sent),dtype=np.int16)
# test_surf_vectors+=dot_surf_index
# for i,sent in enumerate(test_surf_tacy_sents):
#     for j,surf in enumerate(sent):
#         if surf.text in states_map:
#             test_surf_vectors[i,j]=states_map[surf.text]
#         # TODO:Must deal with those who appear in the test, but not in the train type
#         # else:
#         #     sample[i]=

#%%
print('train_lemm_vectors, len: ',len(train_lemm_vectors),'[1]:',train_lemm_vectors[1])
print(train_lemm_tacy_sents[1])
# print(train_surf_tacy_sents[1])
print('test_lemm_vectors, len: ',len(test_lemm_vectors),'[2]:',test_lemm_vectors[2])
print(test_lemm_tacy_sents[2])
# print(test_surf_tacy_sents[2])

#%%
'''
Define the HMM Model
'''
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_= start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#%%
'''
Prediction
'''
print('--- Model HMM predicteur predict on train data:---')
# pred_seqs=[]
# pred_sents=[]
# for i,lemm_seq in enumerate(train_lemm_vectors):
#     # target_surf_origin=test_surf_tacy_sents[2]
#     # X : array-like, shape (n_samples, n_features)
#     # logprob, output_seq = model.decode(input_seq.reshape(-1, 1), algorithm="viterbi")
#     logprob, predict_seq = model.decode(lemm_seq.reshape(-1, 1), algorithm="viterbi")
#     pred_seqs.append(predict_seq)
#     # lemm_seq2sent=decode_sent(lemm_seq,observations)
#     pred_seq2sent=decode_sent(predict_seq,states)
#     pred_sents.append(pred_seq2sent)
#     print('--')
#     print(train_lemm_tacy_sents[i].text)
#     print(train_surf_tacy_sents[i].text)
#     print(pred_seq2sent)
#     if i>5: break;


#%%
print('--- Model HMM predicteur predict on test data: ---')
hmm_pred_seqs=[]
hmm_pred_sents=[]
for i,lemm_seq in enumerate(test_lemm_vectors):
    # target_surf_origin=test_surf_tacy_sents[2]
    # X : array-like, shape (n_samples, n_features)
    # logprob, output_seq = model.decode(input_seq.reshape(-1, 1), algorithm="viterbi/map")
    logprob, predict_seq = model.decode(lemm_seq.reshape(-1, 1), algorithm="viterbi")
    hmm_pred_seqs.append(predict_seq)
    # lemm_seq2sent=decode_sent(lemm_seq,observations)
    pred_seq2sent=decode_sent(predict_seq,states)
    hmm_pred_sents.append(pred_seq2sent)
    print('-- No. ',i)
    print(test_lemm_tacy_sents[i].text)
    print(test_surf_tacy_sents[i].text)
    print(pred_seq2sent)

#%%
hmm_acc_count,count_total=count_accuracy(hmm_pred_sents,test_surf_tacy_sents)
print('Accuracy on HMM predicteur:', hmm_acc_count,'/', count_total,'=',hmm_acc_count/count_total)


