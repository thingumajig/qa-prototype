#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import spacy
from spacy.symbols import *
from udipe_utils import get_udpipe_parser

#Globals
parser_nlp = None

def get_parser_nlp(lang='en'):
  global parser_nlp

  if lang != 'en':
    return get_udpipe_parser(lang)
  elif parser_nlp is None:
    try:
      parser_nlp = spacy.load("en_core_web_sm")
    except:
      parser_nlp = spacy.load("venv\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-2.2.5")

    # parser_nlp.add_pipe(parser_nlp.create_pipe('sentencizer'))

  return parser_nlp



np_labels = {nsubj, nsubjpass, dobj, iobj, pobj}  # Probably others too
np_labels_full = {nsubj, nsubjpass, dobj, iobj, pobj, csubj, csubjpass, attr}  # Probably others too

# print(dir(spacy.symbols))
# subj - subject
# nsubj - nominal subject
# nsubjpass - passive nominal subject
# csubj - clausal subject
# csubjpass - passive clausal subject

def iter_nps(doc):
    for word in doc:
        if word.dep in np_labels_full:
            yield word

def iter_nps_str(doc):
  s = ''
  for np in iter_nps(doc):
    for t in np.subtree:
      s += str(t)+' '
    yield s.strip()
    s = ''


if __name__=='__main__':
    doc = get_parser_nlp()("With respect to all losses caused by the peril of Flood, the Company shall not be liable, in the aggregate for any one Policy year, for more than its proportionate share of US$25,000,000.")
    #doc = get_parser_nlp()("The Program limit of liability is US$400,000,000.")
    print(doc)
    for token in doc:
        print("{2}-{1}({3}-{6}, {0}-{5})".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1))
    print('='*20)
    for np in doc.noun_chunks:
      print(np.text)

    print('='*20)
    for np in iter_nps(doc):
       print(f'{np} -- {np.dep_} {np.tag_}  head: {token.head.text} {token.head.dep_} {token.head.tag_}')
       for t in np.subtree:
         print(f'\t{t}')

    print('='*20)
    for np in iter_nps_str(doc):
      print(np)


    # print('='*20)
    # for token in doc:
    #   print(f'{token.text} {token.tag_} {token.dep_} {str(token.dep)} \t\t\thead: {token.head.tag_} {token.head.dep_} {token.head.text}')     
    #   for t in token.subtree:
    #     print(f'\t{t}')

