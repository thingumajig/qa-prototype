#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from sklearn.cluster import Birch

from text_utils import get_sentences
from tfhubutils2 import *
import numpy as np
from functools import lru_cache


class ExtractiveSummary(object):
  __slots__ = ['text', 'sentences', 'embeddings', 'sentence_encoder']

  def __init__(self, sentence_encoder_name='universal-sentence-encoder-large/5') -> None:
    super().__init__()
    self.sentence_encoder = get_sentence_encoder(sentence_encoder_name)

  def preprocess_text(self, text):
    text = text.strip()
    if self.text != text:
      self.text = text
      self.sentences = get_sentences(self.text)
      print(f'Sentences:{len(self.sentences)}')

      # for s in self.sentences:
      #   self.embeddings +=[e.get_embedding_wo_session()]
      embs = self.sentence_encoder.get_embedding(self.sentences)
      self.embeddings = np.array(embs).tolist()

  def cluster_embeddings(self, threshold=0.4, min_cluster_elements=3, n_clusters = 0):
    if n_clusters == 0:
      n_clusters = None

    bm = Birch(threshold=threshold, n_clusters=n_clusters)
    # bm = Birch(n_clusters=5)
    bm.fit(self.embeddings)

    labels = bm.labels_
    print(f'labels: {labels} [{len(labels)}]')
    centroids = bm.subcluster_centers_
    print(f'centroids:  {len(centroids)} ')
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

  @lru_cache(maxsize=4)
  def get_extractive_texts(self, text, threshold=0.4, min_clusters_elements=3):
    self.preprocess_text(text)
    selected_clusters = self.cluster_embeddings(threshold, min_cluster_elements=min_clusters_elements)

    selected_texts = []
    for s in selected_clusters:
      l = []
      for ns in s[1]:
        if len(self.sentences[ns])>10:
          l.append(self.sentences[ns])

      if len(l)>=min_clusters_elements:
        selected_texts.append((s[0], l))

    return selected_texts


@lru_cache(maxsize=4)
def get_extractive_summary_gen(emb_name='universal-sentence-encoder-multilingual-large/3'):

  return ExtractiveSummary(sentence_encoder_name=emb_name)


if __name__ == '__main__':
  with open('test_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

    es = ExtractiveSummary(sentence_encoder_name='universal-sentence-encoder-multilingual-large/3')

    es.preprocess_text(text)
    es.cluster_embeddings()
