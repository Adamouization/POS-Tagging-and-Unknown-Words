import re

from nltk import FreqDist
from nltk.corpus.util import LazyCorpusLoader

import src.config as config


def download_brown_corpus() -> None:
    """
    Download the Brown corpus and the simplified universal tagset. Store a version locally.
    Note: usually locally stored in ~/nltk_data. Can also be stored in the virtual environment's "lib" or "include"
    directories.
    :return: None.
    """
    import nltk
    nltk.download('brown')
    nltk.download('universal_tagset')


def get_hapax_legomenon(words, unknown_words_threshold: int = 1) -> list:
    """

    :param words:
    :param unknown_words_threshold:
    :return:
    """
    hapax_legomenon = list()
    freq_dist_words = FreqDist(words)
    for i, val in freq_dist_words.items():
        if val <= unknown_words_threshold:
            hapax_legomenon.append(i)
    return hapax_legomenon


def add_start_and_end_of_sentence_tags(data: list) -> list:
    """
    Append <s> and </s> (tagged with their POS) to the training set.
    :param data:
    :return:
    """
    for sentence in data:
        sentence.insert(0, config.START_TAG_TUPLE)
        sentence.insert(len(sentence), config.END_TAG_TUPLE)
    return data


def extract_words(data: list) -> list:
    """

    :param data:
    :return:
    """
    return [w for (w, _) in data]


def extract_tags(data: list) -> list:
    """

    :param data:
    :return:
    """
    return [t for (_, t) in data]


def remove_list_duplicates(data: list) -> list:
    """"""

    return list(set(data))


def get_regex_decimal_number():
    return re.compile(r'\d+(?:,\d*)?')


def print_corpus_information(corpus: LazyCorpusLoader) -> None:
    """

    :param corpus:
    :return:
    """
    print("Number of words in Brown corpus = {}".format(len(corpus.words())))
    print("Number of sentences in Brown corpus = {}".format(len(corpus.tagged_sents(tagset='universal'))))


def print_number_of_sentences(dataset: [list], dataset_name: str) -> None:
    """

    :param dataset:
    :param dataset_name:
    :return:
    """
    counter = sum(x == config.START_TAG_STRING for (x, _) in dataset)
    print("Number of sentences in {}: {}".format(dataset_name, counter))
