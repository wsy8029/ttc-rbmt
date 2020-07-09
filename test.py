import nltk
from nltk.stem import WordNetLemmatizer
sentence = "Turn on the lights whenever pressed the buttons."

nltk.download('wordnet')

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

# Lemmatizing
lm = WordNetLemmatizer()
print([lm.lemmatize(w) for w in tokens])
print([lm.lemmatize(w, pos='v') for w in tokens]) # specified POS