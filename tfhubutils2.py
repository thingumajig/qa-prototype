#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8

import tensorflow_hub as hub

class TFHubContext2:
                         
  def __init__(self, url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3") -> None:
    super().__init__()
    print("initialize model.")

    self.model = hub.load(url)

  def get_embedding(self, texts):
    # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)

    return self.model(texts)

  def close(self):
    print('TFHubContext closed')


from functools import lru_cache

@lru_cache(maxsize=10)
def get_sentence_encoder(name='universal-sentence-encoder-multilingual-large/3'):
  return TFHubContext2(url=f'https://tfhub.dev/google/{name}')

