from nltk.corpus.util import LazyCorpusLoader


def download_brown_corpus() -> None:
    """
    Download the Brown corpus and store a version locally.
    Note: usually locally stored in ~/nltk_data. Can also be stored in the virtual environment's "lib" or "include"
    directories.
    :return: None.
    """
    import nltk
    nltk.download('brown')
    nltk.download('universal_tagset')


def print_corpus_information(corpus: LazyCorpusLoader) -> None:
    print("Number of words in Brown corpus = {}".format(len(corpus.words())))
    print("Number of sentences in Brown corpus = {}".format(len(corpus.tagged_sents(tagset='universal'))))


def print_number_of_sentences(dataset: [list], dataset_name: str) -> None:
    counter = sum(x == "<s>" for (x, _) in dataset)
    print("Number of sentences in {}: {}".format(dataset_name, counter))
