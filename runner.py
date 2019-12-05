# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st
from deeppavlov import build_model, configs
from deeppavlov_utils import get_dp_model

from text_utils import get_sentences

def get_answer(long_text, question):
  return get_dp_model()(long_text, question)

long_text = st.text_area('text:')
question_text = st.text_input('question:')


sentences = get_sentences(long_text)
questions = get_sentences(question_text)

if st.button("Answer"):

  for sent in sentences:
    sents = [sent]*len(questions)
    answers = get_answer(sents, questions)
    # st.write(str(answers))
    if answers[0]:
      head_sentence_writed = False
      for i in range(0, len(answers[0])):
        answer = answers[0][i]
        if answer:

          if not head_sentence_writed:
            st.write(f'<b>sentence:</b> {sent}', unsafe_allow_html=True)
            head_sentence_writed = True

          st.write(f'<b> question:</b> {questions[i]}', unsafe_allow_html=True)
          st.write(f'<b> answer:</b> {answer}', unsafe_allow_html=True)
          # st.write(f'<b>confidence:</b  > {answer[2][0]}', unsafe_allow_html=True)
      if head_sentence_writed:
        st.write(f'<hr>', unsafe_allow_html=True)

    # for question in questions:
    #   answer = get_answer([sent], [question])
    #   # st.write(f'raw debug: {answer}')
    #   if len(answer)>0 and answer[0][0]:
    #     st.write(f'<b>sentence:</b> {sent}', unsafe_allow_html=True)
    #     st.write(f'<b>question:</b> {question}', unsafe_allow_html=True)
    #     st.write(f'<b>answer:</b> {answer[0][0]}', unsafe_allow_html=True)
    #     st.write(f'<b>confidence:</b  > {answer[2][0]}', unsafe_allow_html=True)
    #     st.write(f'<hr>', unsafe_allow_html=True)



  # answer = get_answer(long_text, question)
  # st.write(f'answer: {answer}')
  #
  #
  # st.write(f'sentences: {long_text}')
  # st.write(f'questions: {question}')