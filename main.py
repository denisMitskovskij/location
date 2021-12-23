# coding:utf8
from ipymarkup import show_dep_ascii_markup as show_markup
from razdel import sentenize, tokenize
from navec import Navec
from slovnet import Syntax

text = "В городе президент принял участие в саммите."
chunk = []
for sent in sentenize(text):
    tokens = [_.text for _ in tokenize(sent.text)]
    chunk.append(tokens)
chunk[:1]
[['В', 'городе', 'президент', 'принял', 'участие', 'список', 'в', 'саммите','.']]

navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
syntax = Syntax.load('slovnet_syntax_news_v1.tar')
syntax.navec(navec)

markup = next(syntax.map(chunk))

# Convert CoNLL-style format to source, target indices
words, deps = [], []
for token in markup.tokens:
    words.append(token.text)
    source = int(token.head_id) - 1
    target = int(token.id) - 1
    if source > 0 and source != target:  # skip root, loops
        deps.append([source, target, token.rel])
show_markup(words, deps)

from deeppavlov import build_model, configs
model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)
sentences = ["Я шёл домой по незнакомой улице.", "Девушка пела в церковном хоре."]
for parse in model(sentences):
    print(parse, end="\n\n")