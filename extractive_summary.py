#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from sklearn.cluster import Birch

from text_utils import get_sentences
from tfhubutils2 import *
import numpy as np

class ExtractiveSummary(object):

  __slots__ = ['text', 'sentences', 'embeddings', 'sentence_encoder']

  def __init__(self, text, sentence_encoder_name='universal-sentence-encoder-large/5') -> None:
    super().__init__()
    self.text = text
    self.sentence_encoder = get_sentence_encoder(sentence_encoder_name)


  def preprocess_text(self):
    self.sentences = get_sentences(self.text)
    print(f'Sentences:{len(self.sentences)}')

    # for s in self.sentences:
    #   self.embeddings +=[e.get_embedding_wo_session()]
    embs = self.sentence_encoder.get_embedding(self.sentences)
    self.embeddings = np.array(embs).tolist()


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

