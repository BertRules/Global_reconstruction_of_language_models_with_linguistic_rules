# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:11:19 2020

@author: leoni
"""

from nltk.corpus import wordnet
import nltk

#takes raw sentences and generates the start-dict with tokens and pos-tags for each sentence
def generate_wordnet_dict(tokens_only,postags):
    wordnet_dict = []
    for i,sentence in enumerate(tokens_only):
        wn = {}
        wn['token'] = sentence
        wn['pos'] = postags[i]
        wordnet_dict.append(wn)
    return wordnet_dict
    
#takes one sentence and generates lesk-Synsets for each token of the sentence
def wsd(sentence_token_pos):
    from nltk.wsd import lesk
    lesks = []
    for i,pos in enumerate(sentence_token_pos['pos']):           
            word = sentence_token_pos['token'][i]
            pos = pos[4:]
            if pos in ['VBZ', 'VB', 'VBD', 'VBG', 'VBN','VBP']:
                wn_pos = wordnet.VERB
            elif pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                wn_pos = wordnet.NOUN
            elif pos in ['JJ', 'JJR', 'JJS']:
                wn_pos=wordnet.ADJ
            elif pos in ['RB', 'RBR','RBS']:
                wn_pos=wordnet.ADV
            else:
                wn_pos =''
                word = ''
            lesk_word = lesk(sentence_token_pos['token'],word,wn_pos)
            if lesk_word is None and wn_pos==wordnet.ADV:
                    wn_pos = wordnet.ADJ
                    lesk_word = lesk(sentence_token_pos['token'],word,wn_pos)
            lesks.append(lesk(sentence_token_pos['token'],word,wn_pos))
    return lesks

def find_hypernyms(sentence_lesk):
    hypernyms = []
    for syn in sentence_lesk:
        if syn is not None:
            if not syn.hypernyms():
                hypernyms.append(syn)
            else:
                hypernyms.append(syn.hypernyms())
        else:
            hypernyms.append('O')
    return hypernyms

def find_root_hypernyms(sentence_lesk):
    root_hypernyms =[]
    for syn in sentence_lesk:
        if syn is not None:
            root_hypernyms.append(syn.root_hypernyms())
        else:
            root_hypernyms.append('O')
    return root_hypernyms

def find_holonyms(sentence_lesk):
    holonyms = []
    for syn in sentence_lesk:
        if syn is not None:
            if syn.part_holonyms():
                holonyms.append(syn.part_holonyms())
            else:
                holonyms.append('O')
        else:
            holonyms.append('O')
    return holonyms

def find_antonyms(synsets,pos,tokens):
    from nltk.stem.wordnet import WordNetLemmatizer
    antonyms = []
    for i, syn in enumerate(synsets):
        postag = pos[i][4:]
        #lemmatizing verbs:
        if postag in ['VBZ', 'VB', 'VBD', 'VBG', 'VBN','VBP']:
            word = WordNetLemmatizer().lemmatize(tokens[i],'v')
        else:
            word = tokens[i]
        if str(syn) != 'NoneType':
            string = str(syn)[8:-2] +"."+word
            #print(string)
            try:
                antonyms.append(wordnet.lemma(string).antonyms())
            except:
                antonyms.append('O')
                #print(string)
        else:
            antonyms.append('O')
    return antonyms

    