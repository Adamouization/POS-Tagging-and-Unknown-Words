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


def download_floresta_corpus() -> None:
    """
    Download the portuguese Floresta treebank. Store a version locally.
    Note: usually locally stored in ~/nltk_data. Can also be stored in the virtual environment's "lib" or "include"
    directories.
    :return: None.
    """
    import nltk
    nltk.download('floresta')


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
    Append <s> and </s> (tagged with their POS) to the training set. Removes long sentences (>100 words) from the data.
    :param data: the list of sentences.
    :return: the new list of sentences of length < 100 words with <s> and </s> tags.
    """
    for sentence in data:
        if len(sentence) <= config.MAX_SENTENCE_LENGTH:  # Used to avoid underflow caused by long sentences.
            sentence.insert(0, config.START_TAG_TUPLE)
            sentence.insert(len(sentence), config.END_TAG_TUPLE)
        else:
            data.remove(sentence)
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


def reverse_list(data: list) -> list:
    """
    Reverses a list. Used to place sentence words in order after backtracing.
    :param data: the lit to reverse.
    :return: the reversed list.
    """
    return data[::-1]


def get_regex_decimal_number() -> re.Pattern:
    """
    Regex used to match any decimal number in the string (not using ^ to avoid force matching the entire string).
    Matches decimal numbers with either a ',' or a '.'.
    :return: a regex pattern to extract decimal numbers from a string.
    """
    return re.compile(r'\d+(?:[,.]\d*)?')


def print_corpus_information(corpus: LazyCorpusLoader, corpus_name: str) -> None:
    """
    Prints information about an NLTK corpus e.g. the Brown corpus.
    :param corpus_name:
    :param corpus: the NLTK corpus in use.
    :return: None.
    """
    print("Number of words in {} corpus = {}".format(corpus_name, len(corpus.words())))
    print("Number of sentences in {} corpus = {}".format(corpus_name, len(corpus.tagged_sents(tagset='universal'))))


def print_number_of_sentences(dataset: [list], dataset_name: str) -> None:
    """
    Prints the number of sentences in the dataset.
    :param dataset: a list.
    :param dataset_name: a string to associate the number of sentences to a dataset.
    :return: None.
    """
    counter = sum(x == config.START_TAG_STRING for (x, _) in dataset)
    print("Number of sentences in {}: {}".format(dataset_name, counter))


def print_runtime(runtime):
    """
    Outputs the runtime to the terminal in seconds (with 2 decimals).
    :param runtime: The runtime in seconds.
    :return: None
    """
    print("--- Runtime: {} seconds ---".format(runtime))
