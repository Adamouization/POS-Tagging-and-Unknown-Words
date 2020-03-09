import argparse
from itertools import chain
import os.path
import pickle
from typing import Tuple, Dict, Any

from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.corpus.reader.util import ConcatenatedCorpusView
import pandas as pd

import src.config as config
from src.helpers import *


def main() -> None:
    """
    Program entry point.
    :return: None.
    """
    parse_command_line_arguments()

    # Only need to do this once.
    # download_brown_corpus()
    # print_corpus_information(brown)

    # Retrieve tagged sentences from the Brown corpus
    tagged_sentences = brown.tagged_sents(tagset='universal')

    # Split data into a training and a testing set (default split 95/5 sentences).
    training_set, testing_set = split_train_test_data(tagged_sentences)
    if config.debug:
        print_number_of_sentences(training_set, "training dataset")
        print_number_of_sentences(testing_set, "testing dataset")

    # Store all words and all tags from the training dataset in a ordered lists (and make lists without duplicates).
    training_words = extract_words(training_set)
    unique_training_words = remove_list_duplicates(training_words)
    training_tags = extract_tags(training_set)
    unique_training_tags = remove_list_duplicates(training_tags)

    # Train the POS tagger by generating the tag transition and word emission probability matrices of the HMM.
    tag_transition_probabilities, emission_probabilities = train_tagger(training_set, training_tags)

    # Test the POS tagger on the testing data using the Viterbi back-tracing algorithm.
    test_sentences = list(tagged_sentences[10000:10500])  # todo use a sentence architecture rather than word.
    test_sentences = add_start_and_end_of_sentence_tags(test_sentences)
    test_tagger(testing_set, unique_training_tags, tag_transition_probabilities, emission_probabilities, test_sentences)


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their state in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--recalculate",
                        action="store_true",
                        help="Include this flag to recalculate the HMM's tag transition and word emission probability "
                             "matrices. Otherwise, previously trained versions will be used.")
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Include this flag additional print statements and data for debugging purposes.")
    args = parser.parse_args()
    config.recalculate = args.recalculate
    config.debug = args.debug


def split_train_test_data(sentences: ConcatenatedCorpusView, train_size: int = 10000, test_size: int = 500) \
        -> Tuple[list, list]:
    """

    :param sentences:
    :param train_size:
    :param test_size:
    :return:
    """
    # Create lists for the training/testing sets based on the specified size.
    training_set = list(sentences[:train_size])
    testing_set = list(sentences[train_size:train_size + test_size])

    # Append <s> and </s> (tagged with their POS) to the training and testing set.
    training_set = add_start_and_end_of_sentence_tags(training_set)
    testing_set = add_start_and_end_of_sentence_tags(testing_set)

    # Convert embedded lists into a single list.
    train = list(chain.from_iterable(training_set))
    test = list(chain.from_iterable(testing_set))

    return train, test


def train_tagger(training_set: list, training_tags: list) \
        -> Tuple[Dict[Any, Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    """

    :param training_tags:
    :param training_set:
    :return:
    """
    # Count number of tag transitions and get probabilities of tag transitions.
    # If tag transition probabilities were already calculated, then load them into memory, else calculate them.
    transition_probabilities_file_path = "data_objects/transition_probabilities.pkl"
    if os.path.isfile(transition_probabilities_file_path) and not config.recalculate:
        with open(transition_probabilities_file_path, 'rb') as f:
            tag_transition_probabilities = pickle.load(f)
        print("File '{}' already exists, loaded from memory.".format(transition_probabilities_file_path))
    else:
        transition_occurrences = count_tag_transition_occurrences(training_tags)
        tag_transition_probabilities = get_tag_transition_probability(transition_occurrences)
        with open(transition_probabilities_file_path, 'wb') as f:
            pickle.dump(tag_transition_probabilities, f)

    # Count number of word emissions and get probabilities of emissions.
    # If word emission probabilities were already calculated, then load them into memory, else calculate them.
    emission_probabilities_file_path = "data_objects/emission_probabilities.pkl"
    if os.path.isfile(emission_probabilities_file_path) and not config.recalculate:
        with open(emission_probabilities_file_path, 'rb') as f:
            emission_probabilities = pickle.load(f)
        print("File '{}' already exists, loaded from memory.".format(emission_probabilities_file_path))
    else:
        emission_probabilities = get_emission_probabilities(training_set, training_tags)
        with open(emission_probabilities_file_path, 'wb') as f:
            pickle.dump(emission_probabilities, f)

    # Return the HMMs tag transition and word emission probability matrices.
    return tag_transition_probabilities, emission_probabilities


def count_tag_transition_occurrences(tags: list) -> dict:
    word_list = list()

    # get the list of unique tags
    unique_tags = remove_list_duplicates(tags)

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


# def count_tag_transition_occurrences_new(tags: list) -> dict:
#     tag_transition_occurrences = dict()
#
#     # Loop through each tag in the sentence.
#     current_tag, previous_tag = (str(),) * 2
#     for i, tag in enumerate(tags):
#         if i == 0:  # Special case: first tagged token does not have any preceding token.
#             current_tag = tag
#         else:
#             previous_tag = current_tag
#             current_tag = tag
#
#             # Create a new nested current tag dict in the dictionary.
#             if current_tag not in tag_transition_occurrences.keys():
#                 tag_transition_occurrences[current_tag] = dict()
#
#             # Create a new preceding tag value in the nested dict.
#             if previous_tag not in tag_transition_occurrences[current_tag].keys():
#                 tag_transition_occurrences[current_tag][previous_tag] = 0
#
#             # Increment the occurrence once we are sure that the keys are correctly created in the dict.
#             tag_transition_occurrences[current_tag][previous_tag] += 1
#
#     return tag_transition_occurrences


def get_tag_transition_probability(transitions):
    results_list = list()
    tags = transitions.keys()
    for key in transitions.keys():
        row = {}
        row = transitions[key]["next_tag"]
        row["current_tag"] = key
        results_list.append(row)

    # count_sums = dict()
    # for k, val in transitions.items():
    #     count_sums[k] = 0
    #     for c in val:
    #         count_sums[k] += int(val[c])
    #
    # probabilities = dict()
    # for k, val in transitions.items():
    #     for c in val:
    #         probabilities[k][c] = int(val[c]) / count_sums[k]

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


def get_emission_probabilities(training_set: list, training_tags: list, bins: int = 100000) -> dict:
    """

    :param bins:
    :param training_tags:
    :param training_set:
    :return:
    """
    emission_probabilities = dict()

    tags = remove_list_duplicates(training_tags)
    for tag in tags:
        words = [w for (w, t) in training_set if t == tag]
        emission_probabilities[tag] = WittenBellProbDist(FreqDist(words), bins=bins)

    return emission_probabilities


def test_tagger(testing_set: list, unique_training_tags: list, tag_transition_probabilities: dict,
                emission_probabilities: dict, test_sentences: list) -> None:
    predicted_tags_per_sentence = list()

    # Loop through sentences rather than words to avoid increasingly small probabilities that
    # eventually reach 0 (smallest float value is e-308 in Python).
    for sentence in test_sentences:
        testing_words = [w for (w, _) in sentence]
        # Calculate the Viterbi matrix.
        viterbi_matrix = viterbi_algorithm_smoothed(testing_words, unique_training_tags, tag_transition_probabilities,
                                                    emission_probabilities)
        # Use back-tracing to determine the most likely POS tags for each word in the testing dataset.
        predicted_tags_per_sentence.append(backtrace(viterbi_matrix))
    # Link all lists in predicted_tags_per_sentence into a large single list.
    predicted_tags = list(chain.from_iterable(predicted_tags_per_sentence))

    # viterbi_matrix = viterbi_algorithm_unsmoothed(unique_training_words, unique_training_tags, testing_words,
    # tag_transition_probabilities, emission_probabilities)

    # Print the accuracy of the tagger.
    tagging_accuracy = round(calculate_accuracy(predicted_tags, testing_set), 2)
    print("POS Tagging accuracy on test dataset: {}%".format(tagging_accuracy))


def viterbi_algorithm_unsmoothed(unique_training_words: list, unique_training_tags: list, words: list,
                                 tag_transition_probabilities: dict, emission_probabilities: dict) -> dict:
    # Initialise the Viterbi matrix.
    viterbi_matrix = dict()
    for tag in unique_training_tags:
        viterbi_matrix[tag] = {
            "words": words,
            "viterbi": [0] * len(words)
        }

    # Loop through each word in the testing set.
    for i in range(0, len(words), 1):

        # Check if the word was ever seen in the training set or if it is unknown.
        if viterbi_matrix[tag]["words"][i] in unique_training_words:

            # Special case for first word, (bigram HMMs, don't care about tag before first word). todo: verify reason
            if i == 0:
                continue

            # Special case: # todo verify.
            elif i == 1:
                for tag in viterbi_matrix.keys():
                    current_word = viterbi_matrix[tag]["words"][i]

                    viterbi_matrix[tag]["viterbi"][i] = tag_transition_probabilities["<s>"][tag] * \
                                                        emission_probabilities[current_word][tag]
                    previous_word = viterbi_matrix[tag]["words"][i]

            # All other cases.
            else:
                for tag in viterbi_matrix.keys():
                    temp = list()
                    for k in viterbi_matrix.keys():
                        previous_viterbi = viterbi_matrix[k]["viterbi"][i - 1]
                        val = previous_viterbi * tag_transition_probabilities[k][tag]
                        temp.append(val)
                    max_val = max(temp)

                    current_word = viterbi_matrix[tag]["words"][i]

                    viterbi_matrix[tag]["viterbi"][i] = max_val * emission_probabilities[current_word][tag]

        # Unknown word: never seen in the training set.
        else:
            for tag in viterbi_matrix.keys():
                # Naive handling of unknown words: set the max_val to 1/1000
                viterbi_matrix[tag]["viterbi"][i] = 0.001 * emission_probabilities[current_word][tag]

    return viterbi_matrix


def viterbi_algorithm_smoothed(words: list, unique_training_tags: list, tag_transition_probabilities: dict,
                               emission_probabilities: dict) -> dict:
    # Initialise the Viterbi matrix.
    viterbi_matrix = dict()
    for tag in unique_training_tags:
        viterbi_matrix[tag] = {
            "words": words,
            "viterbi": [0] * len(words)
        }

    # Loop through each word in the testing set.
    for i in range(0, len(words), 1):

        # Special case for first word, (bigram HMMs, don't care about start of sentence tag <s> before the first word.)
        if i == 0:
            continue

        # Special case: # todo verify.
        elif i == 1:
            for tag in viterbi_matrix.keys():
                current_word = viterbi_matrix[tag]["words"][i]
                viterbi_matrix[tag]["viterbi"][i] = tag_transition_probabilities["<s>"][tag] * \
                                                    emission_probabilities[tag].prob(current_word)

        # All other cases.
        else:
            for tag in viterbi_matrix.keys():
                temp = list()
                for k in viterbi_matrix.keys():
                    previous_viterbi = viterbi_matrix[k]["viterbi"][i - 1]
                    val = previous_viterbi * tag_transition_probabilities[k][tag]
                    temp.append(val)
                max_val = max(temp)

                current_word = viterbi_matrix[tag]["words"][i]

                viterbi_matrix[tag]["viterbi"][i] = max_val * emission_probabilities[tag].prob(current_word)

    return viterbi_matrix


def backtrace(viterbi_matrix: dict) -> list:
    """

    :param viterbi_matrix:
    :return:
    """
    backwards_predicted_tags = list()
    sentence_length = len(viterbi_matrix["."]["words"])

    # Start at end of all sentences and back-trace backwards to beginning.
    for i in range(sentence_length - 2, 0, -1):
        predicted_tag = None
        max_viterbi = 0
        for tag in viterbi_matrix.keys():
            viterbi_value = viterbi_matrix[tag]["viterbi"][i]
            if viterbi_value > max_viterbi:
                max_viterbi = viterbi_value
                predicted_tag = tag
        backwards_predicted_tags.append((viterbi_matrix[predicted_tag]["words"][i], predicted_tag))

    # Reverse the predicted tags to get them back in their original order.
    predicted_tags = backwards_predicted_tags[::-1]
    return predicted_tags


def calculate_accuracy(predicted: list, actual: list) -> float:
    """
    Compare the tags predicted from the Viterbi back-tracing algorithm to the tagged version of the testing set.
    :param predicted:
    :param actual:
    :return:
    """
    # Ignore start and end of sentence stags <s> and </s>.
    predicted_trim = list()
    for (w, t) in predicted:
        if w != "<s>" and w != "</s>":
            predicted_trim.append((w, t))
    actual = [(w, t) for (w, t) in actual if w != "<s>" and w != "</s>"]
    # actual_trim = list()
    # for (w, t) in actual:
    #     if w != "<s>" and w != "</s>":
    #         actual_trim.append((w, t))

    # Extract tags only for comparison.
    predicted_tags = extract_tags(predicted_trim)
    actual_tags = extract_tags(actual)

    correct_tag_counter = 0
    for index, (actual_tag, predicted_tag) in enumerate(zip(actual_tags, predicted_tags)):
        if actual_tag == predicted_tag:
            correct_tag_counter += 1

    return (correct_tag_counter / len(actual)) * 100  # Return in the form of a %.


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
#     results_list = list()
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


if __name__ == "__main__":
    main()
