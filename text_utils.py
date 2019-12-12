import spacy
from spacy.pipeline import SentenceSegmenter

nlp = None

def get_nlp():
  global nlp
  if nlp is None:
    try:
      nlp = spacy.load("en_core_web_sm")
    except:
      nlp = spacy.load("venv\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-2.2.5")

    nlp.add_pipe(nlp.create_pipe('sentencizer'))

  return nlp

def parse(text):
  return get_nlp()(text, disable=['parser', "tagger", "ner"])

def get_sentences(text):
  doc = parse(text)
  return get_sentences_from_doc(doc)


def get_sentences_from_doc(doc):
  sentences = []
  for sent in doc.sents:
    s = str(sent).strip()
    if s:
      sentences.append(s)

  # return [str(sent) for sent in doc.sents]
  return sentences


if __name__ == '__main__':

  print(get_sentences(''' Народного артиста СССР Юрия Соломина госпитализировали в Москве. Об этом сообщает РЕН ТВ.

По информации телеканала, 84-летний актер доставлен в больницу в тяжелом состоянии. Он сам вызвал скорую после того, как у него несколько дней держалась высокая температура.
'''))

  