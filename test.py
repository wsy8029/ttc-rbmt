import nltk
from nltk.corpus import treebank
sentence = "Turn on the light whenever press the button."

#nltk.download('treebank')

# Divide tokens by word
tokens = nltk.word_tokenize(sentence)
print(tokens)

# POS tagging each words
tagged = nltk.pos_tag(tokens)
print(tagged)

# NER(Name Entity Recognition) 개체명 인식
# 1. Person 2. Location 3.Organization
entities = nltk.ne_chunk(tagged)
print(entities)

groucho_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
    """)

parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(tagged):
    print(tree)