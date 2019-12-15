# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st

# import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")



import extractive_summary as es
import fancy_cache as fc

# from PIL import Image
#
# image = Image.open('haystac-logo-small.png')
# st.sidebar.image(image, use_column_width=False)


@fc.fancy_cache(unique_to_session=True,allow_output_mutation=True)
def get_extractive_summary_gen():
  return es.create_extractive_summary_gen()

long_text = st.text_area('Text:')

threshold = st.sidebar.slider('threshold:', min_value=0.1, step=0.01, max_value=1., value=0.5)
n_themes = st.sidebar.slider('themes:', min_value=0, step=1, max_value=100, value=0)
min_cluster_elements = st.sidebar.slider('minimum theme sentences:', min_value=1, step=1, max_value=10, value=2)
min_sent_len = st.sidebar.slider('minimum sentence len:', min_value=1, step=1, max_value=300, value=20)
max_naming_len = st.sidebar.slider('maximum name len:', min_value=10, step=1, max_value=300, value=30)


if long_text:
  gen = get_extractive_summary_gen()
  l = gen.get_extractive_texts(long_text, threshold=threshold,
                               min_clusters_elements=min_cluster_elements,
                               n_clusters=n_themes, minimum_sentence_len=min_sent_len)

  if len(l)>0:
    namings = gen.get_cluster_naming(l, max_naming_len = max_naming_len)

    st.subheader('Extactive summary:')
    for theme in l:
      st.subheader(f'Theme: {namings[theme[0]].title()}')
      for sent in theme[1]:
        st.write(sent)
      st.write(f'<hr>', unsafe_allow_html=True)
  else:
    st.write("Nothing found, change threshold")