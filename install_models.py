import os
cmds = [
    'python -m deeppavlov install squad_bert',
    'python -m deeppavlov download deeppavlov\multi_squad_ru_retr_noans_rubert_infer.json',
    'python -m spacy download en_core_web_sm',  #download spacy model
    'python -m spacy download en',
]

for c in cmds:
    os.system(c)
