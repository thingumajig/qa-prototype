#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from sklearn.cluster import Birch

from text_utils import get_sentences
from tfhubutils2 import *
import numpy as np
from functools import lru_cache


class ExtractiveSummary(object):
  __slots__ = ['text', 'sentences', 'embeddings', 'sentence_encoder', 'centroids', 'lang']

  def __init__(self, sentence_encoder_name='universal-sentence-encoder-large/5', lang = 'en') -> None:
    super().__init__()
    self.sentence_encoder = get_sentence_encoder(sentence_encoder_name)
    self.text = None
    self.lang = lang

  def preprocess_text(self, t):
    t = t.strip()
    if self.text != t:
      self.text = t
      self.sentences = get_sentences(self.text)
      print(f'Sentences:{len(self.sentences)}')

      # for s in self.sentences:
      #   self.embeddings +=[e.get_embedding_wo_session()]

      embs = self.sentence_encoder.get_embedding(self.sentences)

      self.embeddings = np.array(embs).tolist()


  def cluster_embeddings(self, threshold=0.4, min_cluster_elements=3, n_clusters=0):
    if n_clusters == 0:
      n_clusters = None

    bm = Birch(threshold=threshold, n_clusters=n_clusters)
    # bm = Birch(n_clusters=5)
    bm.fit(self.embeddings)

    labels = bm.labels_
    print(f'labels: {labels} [{len(labels)}]')
    self.centroids = bm.subcluster_centers_
    print(f'centroids:  {len(self.centroids)} ')
    n_clusters = np.unique(labels).size

    print(f'n_clusters:{n_clusters}')

    from collections import defaultdict

    clusters = defaultdict(list)
    for i, key in enumerate(labels):
      clusters[key].append(i)

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4, width=60, compact=True)
    # pp.pprint(clusters)

    selected_clusters = []
    for k, v in clusters.items():
      if len(v) >= min_cluster_elements:
        selected_clusters.append((k, v))

    # pp.pprint(selected_clusters)

    # print('Themes:')
    # for x in selected_clusters:
    #   print(f'N{x[0]}{"-" * 30}')
    #   for sent in x[1]:
    #     print(es.sentences[sent])
    return selected_clusters

  @lru_cache(maxsize=15)
  def get_extractive_texts(self, t, threshold=0.4, min_clusters_elements=3, n_clusters=0, minimum_sentence_len=20):
    self.preprocess_text(t)
    selected_clusters = self.cluster_embeddings(threshold, min_cluster_elements=min_clusters_elements, n_clusters=n_clusters)

    selected_texts = []
    for s in selected_clusters:
      l = []
      for ns in s[1]:
        if len(self.sentences[ns])>minimum_sentence_len:
          l.append(self.sentences[ns])

      if len(l) >= min_clusters_elements:
        selected_texts.append((s[0], l))

    return selected_texts

  def get_cluster_naming(self, selected_texts, max_naming_len = 30):
    print(f'= Generate Naming {"="*20}')


    from parser_utils import iter_nps_str, get_parser_nlp
    from scipy.spatial import distance

    # select language concept segmenter
    ng_iter = iter_nps_str
    if self.lang == 'ru':
      import udipe_utils as udpu
      ng_iter = udpu.upp_iter_nps_str



    names = dict()
    for x in selected_texts:
      centroid_ix = x[0]
      print(f'centroid idx: {centroid_ix}')
      sents = x[1]
      ngs = set()
      for s in sents:
        ss = get_parser_nlp(lang=self.lang)(s)



        for spart in ng_iter(ss):
          print(f'phrase:{spart} : {len(spart)}')
          if len(spart)<=max_naming_len:
            ngs.add(spart)

      ngs = list(ngs)
      nes = self.sentence_encoder.get_embedding(ngs)
      nes = np.array(nes).tolist()
      centroid = self.centroids[centroid_ix]

      print(f'\tGenerated phrases: {ngs}')

      #argmin
      min_d = 100.  
      min_s = None
      phrases_list = []
      for s, v in zip(ngs, nes):
        d = distance.cosine(v,centroid)
        print(f'\t\t{s} : {d}')
        phrases_list.append((s,d))
        if d < min_d:
          min_d = d
          min_s = s

      names[centroid_ix] = min_s
      print(f'\tSelected phrase: {min_s}')

    return names



def create_extractive_summary_gen(emb_name='universal-sentence-encoder-multilingual-large/3', lang = 'en'):

  return ExtractiveSummary(sentence_encoder_name=emb_name, lang = lang)


if __name__ == '__main__':
  with open('test_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

    es = ExtractiveSummary(sentence_encoder_name='universal-sentence-encoder-multilingual-large/3')

    es.preprocess_text(text)
    es.cluster_embeddings()
