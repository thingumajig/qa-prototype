# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st

import extractive_summary as es
import fancy_cache as fc

@fc.fancy_cache(unique_to_session=True)
def get_extractive_summary_gen():
  return es.create_extractive_summary_gen()

long_text = st.text_area('Text:')

threshold = st.sidebar.slider('threshold:', min_value=0.1, step=0.01, max_value=1., value=0.5)
n_themes = st.sidebar.slider('themes:', min_value=0, step=1, max_value=100, value=0)
min_cluster_elements = st.sidebar.slider('minimum theme sentences:', min_value=1, step=1, max_value=10, value=2)
min_sent_len = st.sidebar.slider('minimum sentence len:', min_value=1, step=1, max_value=300, value=20)


if long_text:
  l = get_extractive_summary_gen().get_extractive_texts(long_text, threshold=threshold,
                                                        min_clusters_elements=min_cluster_elements,
                                                        n_clusters=n_themes, minimum_sentence_len=min_sent_len)

  if len(l)>0:
    st.subheader('Extactive summary:')
    for i, theme in enumerate(l):
      st.subheader(f'Theme #{i+1}')
      for sent in theme[1]:
        st.write(sent)
      st.write(f'<hr>', unsafe_allow_html=True)
  else:
    st.write("Nothing found, change threshold")