from nltk.corpus.util import LazyCorpusLoader


START_TAG_TUPLE = ("<s>", "<s>")
END_TAG_TUPLE = ("</s>", "</s>")


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


def add_start_and_end_of_sentence_tags(data: list) -> list:
    """
    Append <s> and </s> (tagged with their POS) to the training set.
    :param data:
    :return:
    """
    for sentence in data:
        sentence.insert(0, START_TAG_TUPLE)
        sentence.insert(len(sentence), END_TAG_TUPLE)
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
    counter = sum(x == "<s>" for (x, _) in dataset)
    print("Number of sentences in {}: {}".format(dataset_name, counter))
