#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from sklearn.cluster import Birch

from text_utils import get_sentences
from tfhub_utils import *
import numpy as np

class ExtractiveSummary(object):

  __slots__ = ['text', 'sentences', 'embeddings']

  def __init__(self, text) -> None:
    super().__init__()
    self.text = text


  def preprocess_text(self):
    self.sentences = get_sentences(self.text)
    e = get_sentence_encoder()
    # for s in self.sentences:
    #   self.embeddings +=[e.get_embedding_wo_session()]
    embs = e.get_embedding_wo_session(self.sentences)
    self.embeddings = list(embs)

    print(f'Sentences:{len(self.sentences)}')

  def cluster_embeddings(self):
    bm = Birch(n_clusters=None)
    bm.fit(self.embeddings)

    labels = bm.labels_
    centroids = bm.subcluster_centers_
    n_clusters = np.unique(labels).size

    print(f'n_clusters:{n_clusters}')



if __name__=='__main__':
  
  with open('test_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

    es = ExtractiveSummary(str(text))

    es.preprocess_text()
    es.cluster_embeddings()

