from nltk.corpus import brown

from src.helpers import download_brown_corpus


def main():
    sentence = brown.tagged_sents(tagset='universal')
    first = sentence[0]
    print("\nfirst")
    print(first)

    words = [w for (w, _) in first]
    print("\nwords")
    print(words)

    tags = [t for (_, t) in first]
    print("\ntags")
    print(tags)
    for s in sentence[0:10]:
        show_sent(s)


def show_sent(sentence):
    print("\nsentence")
    print(sentence)


if __name__ == "__main__":
    main()
