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
    Retrieve a list of words occurring no more than "unknown_words_threshold", which corresponds to the number of words
    occurring once in a dataset.
    :param words: list of words.
    :param unknown_words_threshold: observation frequency of words.
    :return: list of hapax legomenon.
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
    Extract words from the tokens of a sentence from the data.
    :param data:
    :return:
    """
    words = list()
    for sentence in data:
        for (w, _) in sentence:
            words.append(w)
    return words


def extract_tags(data: list) -> list:
    """
    Extract tags from the tokens of a sentence from the data.
    :param data: a list of sentence (list of tokens)
    :return: a list of tags
    """
    tags = list()
    for sentence in data:
        for (_, t) in sentence:
            tags.append(t)
    return tags


def remove_list_duplicates(data: list) -> list:
    """
    Removes duplicate values from a list by converting it to a set and then back to a list.
    :param data: the data to remove the duplicates from.
    :return: a list of unique values.
    """
    return list(set(data))


def get_regex_decimal_number() -> re.Pattern:
    """
    Regex used to match any decimal number in the string.
    Source: https://stackoverflow.com/a/44507054/5609328.
    :return: a regex to extract decimal numbers from a string.
    """
    return re.compile(r'\d+(?:,\d*)?')


def print_corpus_information(corpus: LazyCorpusLoader) -> None:
    """
    Prints information about an NLTK corpus e.g. the Brown corpus.
    :param corpus: the NLTK corpus in use.
    :return: None.
    """
    print("Number of words in Brown corpus = {}".format(len(corpus.words())))
    print("Number of sentences in Brown corpus = {}".format(len(corpus.tagged_sents(tagset='universal'))))


def print_number_of_sentences(dataset: [list], dataset_name: str) -> None:
    """
    Prints the number of sentences in the dataset.
    :param dataset: a list.
    :param dataset_name: a string to associate the number of sentences to a dataset.
    :return: None.
    """
    counter = sum(x == config.START_TAG_STRING for (x, _) in dataset)
    print("Number of sentences in {}: {}".format(dataset_name, counter))
