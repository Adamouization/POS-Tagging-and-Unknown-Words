import argparse
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
    Program entry point. Main execution flow of the POS tagger:
        - load corpus data
        - split data into train/testing sets
        - handle unknown words in both sets
        - train tagger (calculate HMM's word emission and tag transision probabilities)
        - test tagger (Viberbi algorithm, backtracing and accuracy measure)
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

    # Replace infrequent words with special 'UNK' tags.
    training_words = extract_words(training_set)
    unique_training_words = remove_list_duplicates(training_words)
    training_set = handle_unknown_words(training_set, unique_training_words, is_training_set=True)
    testing_set = handle_unknown_words(testing_set, unique_training_words, is_training_set=False)

    # Store all words and all tags from the training dataset in a ordered lists (and make lists without duplicates).
    training_tags = extract_tags(training_set)
    unique_training_tags = remove_list_duplicates(training_tags)

    # Train the POS tagger by generating the tag transition and word emission probability matrices of the HMM.
    tag_transition_probabilities, emission_probabilities = train_tagger(training_set, training_tags)

    # Test the POS tagger on the testing data using the Viterbi back-tracing algorithm.
    test_tagger(testing_set, unique_training_tags, tag_transition_probabilities, emission_probabilities)


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


def split_train_test_data(sentences: ConcatenatedCorpusView, train_size: int = config.DEFAULT_TRAIN_SIZE,
                          test_size: int = config.DEFAULT_TEST_SIZE) -> Tuple[list, list]:
    """
    Splits the dataset into a training and a testing dataset.
    :param sentences: the corpus' list of sentences.
    :param train_size: the size of the training set.
    :param test_size: the size of the testing set.
    :return: a tuple containing both split data sets.
    """
    # Create lists for the training/testing sets based on the specified size.
    training_set = list(sentences[:train_size])
    testing_set = list(sentences[train_size:train_size + test_size])

    # Append <s> and </s> (tagged with their POS) to the training and testing set.
    training_set = add_start_and_end_of_sentence_tags(training_set)
    testing_set = add_start_and_end_of_sentence_tags(testing_set)

    return training_set, testing_set


def handle_unknown_words(dataset: list, unique_training_words: list, is_training_set: bool = True):
    """
    Checks for infrequent words and replaces them with appropriate "UNK-x" strings. In the testing set, checks that the
    infrequent word in question is not in the training set.
    :param dataset: the list of sentences.
    :param unique_training_words: the list words without duplicates.
    :param is_training_set: a boolean to differentiate whether the set is the training or the testing set.
    :return: the updated dataset.
    """
    # Store all words from the dataset in an ordered list.
    words = extract_words(dataset)

    # Get the hapax legomenon of the dataset and replace the words in the training set with the new 'UNK' words.
    if is_training_set:
        dataset = replace_training_words(dataset, get_hapax_legomenon(words))
    # Replace the words in the the testing set with the new 'UNK' words.
    else:
        dataset = replace_testing_words(dataset, unique_training_words)

    return dataset


def replace_training_words(dataset: list, hapax_legomenon: list) -> list:
    """

    :param dataset:
    :param hapax_legomenon:
    :return:
    """
    for i, sentence in enumerate(dataset):
        for j in range(0, len(sentence)):
            if sentence[j][0] in hapax_legomenon:
                dataset[i][j] = (unknown_words_rules(sentence[j][0]), dataset[i][j][1])
    return dataset


def replace_testing_words(dataset: list, unique_training_words: list) -> list:
    """

    :param dataset:
    :param unique_training_words:
    :return:
    """
    for i, sentence in enumerate(dataset):
        for j in range(0, len(sentence)):
            if sentence[j][0] not in unique_training_words:
                dataset[i][j] = (unknown_words_rules(sentence[j][0]), dataset[i][j][1])
    return dataset


def unknown_words_rules(word: str) -> str:
    """

    :param word:
    :return:
    """
    if word.startswith('$'):
        return "UNK-currency"
    elif word.endswith('ed'):
        return "UNK-ed"
    elif word.endswith('ing'):
        return"UNK-ing"
    elif word.endswith("'s"):
        return"UNK-apostrophe-s"
    elif word.istitle():
        return "UNK-uppercase"
    elif word.isdigit():
        return "UNK-number"
    elif get_regex_decimal_number().match(word):
        return "UNK-decimal-number"
    elif '-' in word:
        return "UNK-hyphen"
    return "UNK"


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
        # tag_transition_probabilities = get_tag_transition_probabilities(training_set, training_tags)
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


def count_tag_transition_occurrences_new(tags: list) -> dict:
    tag_transition_occurrences = dict()

    # Loop through each tag in the sentence.
    current_tag, previous_tag = (str(),) * 2
    for i, tag in enumerate(tags):
        if i == 0:  # Special case: first tagged token does not have any preceding token.
            current_tag = tag
        else:
            previous_tag = current_tag
            current_tag = tag

            # Create a new nested current tag dict in the dictionary.
            if current_tag not in tag_transition_occurrences.keys():
                tag_transition_occurrences[current_tag] = dict()

            # Create a new preceding tag value in the nested dict.
            if previous_tag not in tag_transition_occurrences[current_tag].keys():
                tag_transition_occurrences[current_tag][previous_tag] = 0

            # Increment the occurrence once we are sure that the keys are correctly created in the dict.
            tag_transition_occurrences[current_tag][previous_tag] += 1

    return tag_transition_occurrences


def get_tag_transition_probabilities(training_set: list, training_tags: list, bins: int = 100) -> dict:
    transition_probabilities = dict()
    tags = remove_list_duplicates(training_tags)

    tag_pairs = count_tag_transition_occurrences_new(training_tags)

    for tag in tags:
        transition_probabilities[tag] = WittenBellProbDist(FreqDist(tag_pairs[tag]), bins=bins)

    for tag in tags:
        prev_tags = list()
        for sentence in training_set:
            for i in range(1, len(sentence)):
                prev_tags.append(sentence[i-1][1])
        transition_probabilities[tag] = FreqDist(prev_tags)
        pass
        # transition_probabilities[tag] = WittenBellProbDist(FreqDist(prev_tags), bins=bins)

    return transition_probabilities


def get_tag_transition_probability(transitions):
    # count_sums = dict()
    # for cur_tag, prev_tags in transitions.items():
    #     count_sums[cur_tag] = 0
    #     for prev_tag in prev_tags:
    #         count_sums[cur_tag] += int(prev_tags[prev_tag])
    #
    # probabilities = dict()
    # for cur_tag, prev_tags in transitions.items():
    #     # if cur_tag not in probabilities.keys():
    #     for prev_tag in prev_tags:
    #         val = int(prev_tags[prev_tag]) / count_sums[cur_tag]
    #         probabilities[cur_tag][prev_tag] = val

    results_list = list()
    tags = transitions.keys()
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


def get_emission_probabilities(training_set: list, training_tags: list, bins: int = 100000000) -> dict:
    """

    :param bins:
    :param training_tags:
    :param training_set:
    :return:
    """
    emission_probabilities = dict()

    tags = remove_list_duplicates(training_tags)
    for tag in tags:
        words = list()
        for sentence in training_set:
            for (w, t) in sentence:
                if t == tag:
                    words.append(w)
        emission_probabilities[tag] = WittenBellProbDist(FreqDist(words), bins=bins)

    return emission_probabilities


def test_tagger(testing_set: list, unique_training_tags: list, tag_transition_probabilities: dict,
                emission_probabilities: dict) -> None:
    predicted_tags = list()
    # Loop through sentences rather than words to avoid increasingly small probabilities that
    # eventually reach 0 (smallest float value is e-308 in Python).
    for sentence in testing_set:
        testing_words = [w for (w, _) in sentence]
        # Calculate the Viterbi matrix.
        viterbi_matrix = viterbi_algorithm_smoothed(
            testing_words,
            unique_training_tags,
            tag_transition_probabilities,
            emission_probabilities
        )
        # Use back-tracing to determine the most likely POS tags for each word in the testing dataset.
        predicted_tags.append(backtrace(viterbi_matrix))

    # viterbi_matrix = viterbi_algorithm_unsmoothed(unique_training_words, unique_training_tags, testing_words,
    # tag_transition_probabilities, emission_probabilities)

    # Print the accuracy of the tagger.
    tagging_accuracy = round(calculate_accuracy(predicted_tags, testing_set), 2)
    print("POS Tagging accuracy on test dataset: {}%".format(tagging_accuracy))


def viterbi_algorithm_smoothed(words: list, unique_training_tags: list, tag_transition_probabilities: dict,
                               emission_probabilities: dict) -> dict:
    # Initialise the Viterbi matrix.
    viterbi_matrix = dict()
    for tag in unique_training_tags:
        viterbi_matrix[tag] = {
            "words": words,
            "viterbi_value": [0] * len(words)
        }

    # Loop through each word in the testing set.
    for i in range(0, len(words)):

        # Special case for first word, (bigram HMMs, don't care about start of sentence tag <s> before the first word.)
        if i == 0:
            continue

        # Special case: previous tag is <s>.
        elif i == 1:
            for tag in viterbi_matrix.keys():
                current_word = viterbi_matrix[tag]["words"][i]
                viterbi_matrix[tag]["viterbi_value"][i] = tag_transition_probabilities[config.START_TAG_STRING][tag] * \
                                                          emission_probabilities[tag].prob(current_word)
                # viterbi_matrix[tag]["viterbi_value"][i] = tag_transition_probabilities[config.START_TAG_STRING].prob(tag) * \
                #                                           emission_probabilities[tag].prob(current_word)

        # All other cases.
        else:
            for tag in viterbi_matrix.keys():
                cur_tag_viterbi_values = list()
                for key in viterbi_matrix.keys():
                    previous_viterbi_value = viterbi_matrix[key]["viterbi_value"][i - 1]
                    cur_viterbi_value = previous_viterbi_value * tag_transition_probabilities[key][tag]
                    # cur_viterbi_value = previous_viterbi_value * tag_transition_probabilities[key].prob(tag)
                    cur_tag_viterbi_values.append(cur_viterbi_value)
                max_viterbi = max(cur_tag_viterbi_values)
                viterbi_matrix[tag]["viterbi_value"][i] = \
                    max_viterbi * emission_probabilities[tag].prob(viterbi_matrix[tag]["words"][i])

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
            viterbi_value = viterbi_matrix[tag]["viterbi_value"][i]
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
    # Extract tags from sentence tokens (ang ignore start and end of sentence stags <s> and </s>).
    predicted_tags = list()
    for sentence in predicted:
        for (w, t) in sentence:
            if w != config.START_TAG_STRING and w != config.END_TAG_STRING:
                predicted_tags.append(t)
    actual_tags = list()
    for sentence in actual:
        for (w, t) in sentence:
            if w != config.START_TAG_STRING and w != config.END_TAG_STRING:
                actual_tags.append(t)

    # Count number of correct predictions.
    correct_tag_counter = 0
    for i in range(0, len(actual_tags)):
        if actual_tags[i] == predicted_tags[i]:
            correct_tag_counter += 1

    # Calculate percentage of correct predictions.
    return (correct_tag_counter / len(actual_tags)) * 100


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

# def viterbi_algorithm_unsmoothed(unique_training_words: list, unique_training_tags: list, words: list,
#                                  tag_transition_probabilities: dict, emission_probabilities: dict) -> dict:
#     # Initialise the Viterbi matrix.
#     viterbi_matrix = dict()
#     for tag in unique_training_tags:
#         viterbi_matrix[tag] = {
#             "words": words,
#             "viterbi": [0] * len(words)
#         }
#
#     # Loop through each word in the testing set.
#     for i in range(0, len(words), 1):
#
#         # Check if the word was ever seen in the training set or if it is unknown.
#         if viterbi_matrix[tag]["words"][i] in unique_training_words:
#
#             # Special case for first word, (bigram HMMs, don't care about tag before first word). todo: verify reason
#             if i == 0:
#                 continue
#
#             # Special case: # todo verify.
#             elif i == 1:
#                 for tag in viterbi_matrix.keys():
#                     current_word = viterbi_matrix[tag]["words"][i]
#
#                     viterbi_matrix[tag]["viterbi"][i] = tag_transition_probabilities[config.START_TAG_STRING][tag] * \
#                                                         emission_probabilities[current_word][tag]
#                     previous_word = viterbi_matrix[tag]["words"][i]
#
#             # All other cases.
#             else:
#                 for tag in viterbi_matrix.keys():
#                     temp = list()
#                     for k in viterbi_matrix.keys():
#                         previous_viterbi = viterbi_matrix[k]["viterbi"][i - 1]
#                         val = previous_viterbi * tag_transition_probabilities[k][tag]
#                         temp.append(val)
#                     max_val = max(temp)
#
#                     current_word = viterbi_matrix[tag]["words"][i]
#
#                     viterbi_matrix[tag]["viterbi"][i] = max_val * emission_probabilities[current_word][tag]
#
#         # Unknown word: never seen in the training set.
#         else:
#             for tag in viterbi_matrix.keys():
#                 # Naive handling of unknown words: set the max_val to 1/1000
#                 viterbi_matrix[tag]["viterbi"][i] = 0.001 * emission_probabilities[current_word][tag]
#
#     return viterbi_matrix


if __name__ == "__main__":
    main()
