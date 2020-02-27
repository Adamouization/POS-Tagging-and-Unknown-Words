def download_brown_corpus():
    """
    Download the Brown corpus and store a version locally.
    :return: None.
    """
    import nltk
    nltk.download('brown')


def print_corpus_information(corpus):
    print("Number of words in Brown corpus = {}".format(len(corpus.words())))
    print("Number of sentences in Brown corpus = {}".format(len(corpus.tagged_sents(tagset='universal'))))
