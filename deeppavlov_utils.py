# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st
from deeppavlov import build_model, configs

model = None


def get_dp_model():
    global model
    if model is None:
        #model = build_model(configs.squad.squad, download=True)
        model = build_model(
            configs.squad.multi_squad_ru_retr_noans_rubert_infer,
            download=False)
        #model = build_model(configs.squad.squad_bert_infer, download=True)
    return model
