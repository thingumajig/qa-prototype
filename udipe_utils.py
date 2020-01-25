#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import pymorphy2
import spacy_udpipe

from spacy.symbols import *

import nltk_tree as ntree

udpipe_nlp = dict()

def get_udpipe_parser(lang='ru'):
  global udpipe_nlp
  if udpipe_nlp.get(lang, None) is None:
    try:

      spacy_udpipe.download(lang)  # download Russian model

      udpipe_nlp[lang] = spacy_udpipe.load("ru")
      # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    except:
      print(f'error loading udpipe model for {lang}')

    # parser_nlp.add_pipe(parser_nlp.create_pipe('sentencizer'))

  return udpipe_nlp[lang]




np_labels_full = {nsubj, nsubjpass, dobj, iobj, pobj, csubj, csubjpass, attr,
                  obj, nmod}  # obl, nmod,  Probably others too

def iter_nps(doc):
  for word in doc:
    if word.dep in np_labels_full:
      yield word


exclude_labels = {nmod, acl}


def upp_iter_nps_str(doc):
  s = ''
  for np in iter_nps(doc):

    excluded = set()
    for child in np.children:
      if child is not np and child.dep in exclude_labels:
        excluded.add(child)
        for t in child.subtree:
          excluded.add(t)
      elif child.dep == 8110129090154140942:  # child.dep_ == case
        excluded.add(child)

    for t in np.subtree:
      # if t.head.dep == nmod
      if t not in excluded:
        s += str(t.lemma_) + ' '
    yield s.strip()
    s = ''


if __name__ == '__main__':

  text = '''
  Сухопутные войска Великобритании в декабре 2019 года испытали одну из модификаций танка Challenger 2 (Streetfighter II) центре боевой подготовки в городских условиях Коупхилл-Даун на военном полигоне в Солсбери (Англия), сообщает Jane's Defence Weekly. Соответствующий ролик выложен на YouTube.
  Британский журнал отмечает, что городской танк Streetfighter II получил распределенную по его корпусу систему инфракрасных и электрооптических датчиков IronVision израильской компании Elbit Systems, позволяющую отображать оперативную обстановку вокруг боевой машины на нашлемный дисплей танкиста.
  '''
  #TODO
  s1 = 'одобрение сделок с недвижимостью'
  s2 = 'утверждение сделок с имуществом'
  s3 = 'одобрение сделок за исключением сделок с недвижимостью'

  doc = get_udpipe_parser()(text, disable=['ner'])

  print('Sentences:' + '=' * 10)
  for sent in doc.sents:
    s = str(sent).strip()
    print(s)

  print('Tokens:' + '=' * 10)
  for token in doc:
    # print(f'{token.text}\t{token.lemma_}\t{token.pos_}\t{token.dep_}:{token.dep}')
    s = ''
    cc = 0
    for t in token.subtree:
      s += str(t) + ' '
      cc += 1
    # if cc>1:
    print(f'[{token.dep_}-{token.dep}-{token.pos_}--{token.text}]\t{s}')
    if token.n_lefts + token.n_rights > 0:
      ntree.print_tree(token)

  print('=' * 20)

  for np in upp_iter_nps_str(doc):
    print(np)


