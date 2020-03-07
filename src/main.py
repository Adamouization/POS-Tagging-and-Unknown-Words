from itertools import chain
import os.path
import pickle
from typing import Tuple, Dict, Any

from nltk.corpus import brown
from nltk.corpus.reader.util import ConcatenatedCorpusView
import pandas as pd

from src.helpers import *


# Constants
START_TAG_TUPLE = ("<s>", "<s>")
END_TAG_TUPLE = ("</s>", "</s>")


def main() -> None:
    """
    Program entry point.
    :return: None.
    """
    # Only need to do this once.
    # download_brown_corpus()

    # Retrieve tagged sentences from the Brown corpus
    tagged_sentences = brown.tagged_sents(tagset='universal')

    # Split data into a training and a testing set (default split 95/5 sentences).
    training_set, testing_set = split_train_test_data(tagged_sentences)
    print_number_of_sentences(training_set, "training dataset")
    print_number_of_sentences(testing_set, "testing dataset")

    tag_transition_probabilities, emission_probabilities = train_tagger(training_set)

    test_tagger()


def split_train_test_data(sentences: ConcatenatedCorpusView, train_size: int = 10000, test_size: int = 500) \
        -> Tuple[list, list]:
    training_set = list(sentences[:train_size])
    test_set = list(sentences[train_size:train_size + test_size])

    # Append <s> and </s> (tagged with their POS) to the training set.
    for sentence in training_set:
        sentence.insert(0, START_TAG_TUPLE)
        sentence.insert(len(sentence), END_TAG_TUPLE)

    # Append <s> and </s> (tagged with their POS) to the testing set.
    for sentence in test_set:
        sentence.insert(0, START_TAG_TUPLE)
        sentence.insert(len(sentence), END_TAG_TUPLE)

    training_words = list(chain.from_iterable(training_set))
    test_words = list(chain.from_iterable(test_set))

    return training_words, test_words


def train_tagger(training_set: list) -> Tuple[Dict[Any, Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    # Store all words and all tags from the training dataset in a single ordered list.
    training_words = [w for (w, _) in training_set]
    training_tags = [t for (_, t) in training_set]

    # Count number of tag transitions.
    transition_occurrences_file_path = "data_objects/transition_occurences.pkl"
    if os.path.isfile(transition_occurrences_file_path):
        with open(transition_occurrences_file_path, 'rb') as f:
            transition_occurences = pickle.load(f)
        print("File '{}' already exists, loaded from memory.".format(transition_occurrences_file_path))
    else:
        transition_occurences = count_tag_transition_occurrences(training_tags)
        with open(transition_occurrences_file_path, 'wb') as f:
            pickle.dump(transition_occurences, f)

    # Get probabilities of tag transitions.
    tag_transition_probabilities = get_tag_transition_probability(transition_occurences)

    # Count number of emissions.
    emission_occurrences_file_path = "data_objects/emission_occurences.pkl"
    if os.path.isfile(emission_occurrences_file_path):
        with open(emission_occurrences_file_path, 'rb') as f:
            emission_occurences = pickle.load(f)
        print("File '{}' already exists, loaded from memory.".format(emission_occurrences_file_path))
    else:
        emission_occurences = count_emission_occurrences(training_words, training_tags)
        with open(emission_occurrences_file_path, 'wb') as f:
            pickle.dump(emission_occurences, f)

    # Get probabilities of emissions.
    emission_probabilities = get_emission_probabilities(emission_occurences)

    return tag_transition_probabilities, emission_probabilities


def test_tagger():
    pass


def count_tag_transition_occurrences(tags: list) -> dict:
    word_list = []

    # get the list of unique tags
    unique_tags = set(tags)

    ## create a dictionary current_tag whose keys are the unique tags
    ## Each value in current_tag is also a dict D, whose keys are the counts of those tags
    ## Initialise the counts to 0
    current_tags = {}
    for i in unique_tags:
        current_tags[i] = {}
        current_tags[i]["current_tag"] = i
        current_tags[i]["next_tag"] = {}

        ## Loop through and initialise the counts to 0
        for j in unique_tags:
            current_tags[i]["next_tag"][j] = 0

    # Loop through the list of tags provided in the argument of this function
    ## and increment the corresponding value in current_tag based on the observation of
    ## the tag of the next word. This provides the transitions counts
    for index, val in enumerate(tags):
        if index != len(tags) - 1:
            current_tags[val]["next_tag"][tags[index + 1]] += 1

    return current_tags


def get_tag_transition_probability(transitions):
    results_list = []
    for key in transitions.keys():
        row = {}
        row = transitions[key]["next_tag"]
        row["current_tag"] = key
        results_list.append(row)

    transitions_df = pd.DataFrame.from_dict(results_list)
    transitions_df["row_total"] = transitions_df.iloc[:, 0:14].sum(axis=1)
    transitions_df = transitions_df[['current_tag', 'X', 'PRON', 'ADP', 'VERB', 'NOUN', 'DET', '.', 'ADV', 'PRT', 'ADJ',
                                     'CONJ', 'NUM', "<s>", "</s>", "row_total"]]

    cols = transitions_df.columns
    cols = [c for c in cols if c != "current_tag" and c != "row_total"]

    new_cols = ["current_tag"]

    for c in cols:
        col_name = c + "_pct"
        new_cols.append(col_name)
        transitions_df[col_name] = transitions_df[c] / transitions_df["row_total"]

    transitions_dict = {}
    for index, row in transitions_df[new_cols].iterrows():
        probabilities = {}
        for i in new_cols[1:]:
            j = i.split("_")[0]
            probabilities[j] = row[i]

        transitions_dict[row["current_tag"]] = probabilities

    return transitions_dict


def count_emission_occurrences(words: list, tags: list) -> dict:
    unique_words = list(set(words))
    unique_tags = list(set(tags))

    current_words = {}

    for i in unique_words:
        current_words[i] = {}
        current_words[i]["tag"] = {}

        for j in unique_tags:
            current_words[i]["tag"][j] = 0

    for index, (w, p) in enumerate(zip(words, tags)):
        current_words[w]["tag"][p] += 1

    return current_words


def get_emission_probabilities(emissions):
    results_list = []
    for key in emissions.keys():
        row = {}
        row = emissions[key]["tag"]
        row["current_word"] = key
        results_list.append(row)

    transitions_df = pd.DataFrame.from_dict(results_list)
    transitions_df["row_total"] = transitions_df.iloc[:, 0:14].sum(axis=1)
    transitions_df = transitions_df[
        ['current_word', 'X', 'PRON', 'ADP', 'VERB', 'NOUN', 'DET', '.', 'ADV', 'PRT', 'ADJ',
         'CONJ', 'NUM', "<s>", "</s>", "row_total"]]

    cols = transitions_df.columns
    cols = [c for c in cols if c != "current_word" and c != "row_total"]

    new_cols = ["current_word"]

    for c in cols:
        col_name = c + "_pct"
        new_cols.append(col_name)
        transitions_df[col_name] = transitions_df[c] / transitions_df["row_total"]

    transitions_dict = {}
    for index, row in transitions_df[new_cols].iterrows():
        probabilities = {}
        for i in new_cols[1:]:
            j = i.split("_")[0]
            probabilities[j] = row[i]

        transitions_dict[row["current_word"]] = probabilities

    return transitions_dict


# def count_tag_transition_occurrences(tagged_sentences: list) -> dict:
#     tag_transition_occurrences = dict()
#
#     # Loop through each sentence in the Brown corpus.
#     for sentence in tagged_sentences:
#
#         # Collect all tags in the sentence.
#         tags = [t for (_, t) in sentence]
#
#         # Loop through each tag in the sentence.
#         current_tag, previous_tag = (str(),) * 2
#         for i, tag in enumerate(tags):
#             if i == 0:  # Special case: first tagged token does not have any preceding token.
#                 current_tag = tag
#             else:
#                 previous_tag = current_tag
#                 current_tag = tag
#
#                 # Create a new nested current tag dict in the dictionary.
#                 if current_tag not in tag_transition_occurrences.keys():
#                     tag_transition_occurrences[current_tag] = dict()
#
#                 # Create a new preceding tag value in the nested dict.
#                 if previous_tag not in tag_transition_occurrences[current_tag].keys():
#                     tag_transition_occurrences[current_tag][previous_tag] = 0
#
#                 # Increment the occurrence once we are sure that the keys are correctly created in the dict.
#                 tag_transition_occurrences[current_tag][previous_tag] += 1
#
#     return tag_transition_occurrences
#
#
# def get_tag_transition_probability(transitions):
#     results_list = []
#     for key in transitions.keys():
#         row = transitions[key]
#         row["current_tag"] = key
#         results_list.append(row)
#
#     transitions_df = pd.DataFrame.from_dict(results_list)
#     transitions_df["row_total"] = transitions_df.iloc[:, 0:14].sum(axis=1)
#     transitions_df = transitions_df[['current_tag', 'X', 'PRON', 'ADP', 'VERB', 'NOUN', 'DET', '.', 'ADV', 'PRT', 'ADJ', 'CONJ', 'NUM', "<s>", "</s>", "row_total"]]
#
#     cols = transitions_df.columns
#     cols = [c for c in cols if c != "current_tag" and c != "row_total"]
#
#     new_cols = ["current_tag"]
#
#     for c in cols:
#         col_name = c + "_pct"
#         new_cols.append(col_name)
#         transitions_df[col_name] = transitions_df[c] / transitions_df["row_total"]
#
#     transitions_dict = {}
#     for index, row in transitions_df[new_cols].iterrows():
#         probabilities = {}
#         for i in new_cols[1:]:
#             j = i.split("_")[0]
#             probabilities[j] = row[i]
#
#         transitions_dict[row["current_tag"]] = probabilities
#
#     return transitions_dict
#
#
# def count_emission_occurences(tagged_sentences: ConcatenatedCorpusView) -> dict:
#     word_tag_pairs_occurrences = dict()
#
#     # Loop through each sentence in the Brown corpus.
#     for sentence in tagged_sentences:
#
#         # Loop through each tagged word in the sentence.
#         for tagged_word in sentence:
#             word = tagged_word[0]
#             tag = tagged_word[1]
#
#             # Create a new nested current tag dict in the dictionary.
#             if word not in word_tag_pairs_occurrences.keys():
#                 word_tag_pairs_occurrences[word] = dict()
#
#             # Create a new preceding tag value in the nested dict.
#             if tag not in word_tag_pairs_occurrences[word].keys():
#                 word_tag_pairs_occurrences[word][tag] = 0
#
#             # Increment the occurrence once we are sure that the keys are correctly created in the dict.
#             word_tag_pairs_occurrences[word][tag] += 1
#
#     return word_tag_pairs_occurrences
#
#
# def get_emission_probabilities(emissions):
#     results_list = []
#     for key in emissions.keys():
#         row = {}
#         row = emissions[key]["tag"]
#         row["current_word"] = key
#         results_list.append(row)
#
#     transitions_df = pd.DataFrame.from_dict(results_list)
#     transitions_df["row_total"] = transitions_df.iloc[:, 0:14].sum(axis=1)
#     transitions_df = transitions_df[['current_word', 'X', 'PRON', 'ADP', 'VERB', 'NOUN', 'DET', '.', 'ADV', 'PRT',
#                                      'ADJ', 'CONJ', 'NUM', "<s>", "</s>", "row_total"]]
#
#     cols = transitions_df.columns
#     cols = [c for c in cols if c != "current_word" and c != "row_total"]
#
#     new_cols = ["current_word"]
#
#     for c in cols:
#         col_name = c + "_pct"
#         new_cols.append(col_name)
#         transitions_df[col_name] = transitions_df[c] / transitions_df["row_total"]
#
#     transitions_dict = {}
#     for index, row in transitions_df[new_cols].iterrows():
#         probabilities = {}
#         for i in new_cols[1:]:
#             j = i.split("_")[0]
#             probabilities[j] = row[i]
#
#         transitions_dict[row["current_word"]] = probabilities
#
#     return transitions_dict


if __name__ == "__main__":
    main()
