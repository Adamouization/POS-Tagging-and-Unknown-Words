from nltk.corpus import brown

import nltk
nltk.download('brown')

sents = brown.tagged_sents(tagset='universal')
first = sents[0]
print(first)
words = [w for (w, _) in first]
print(words)
tags = [t for (_, t) in first]
print(tags)


def show_sent(sent):
    print(sent)


for sent in sents[0:10]:
    show_sent(sent)
