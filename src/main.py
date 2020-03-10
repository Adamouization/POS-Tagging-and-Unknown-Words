import argparse
import os.path
import pickle
import time
from typing import Tuple, Dict, Any

from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown, floresta
from nltk.corpus.reader.util import ConcatenatedCorpusView

import src.config as config
from src.helpers import *


def main() -> None:
    """
    Program entry point. Main execution flow of the POS tagger:
        - load corpus data
        - split data into train/testing sets
        - handle unknown words in both sets
        - train tagger (calculate HMM's word emission and tag transition probabilities)
        - test tagger (Viberbi algorithm, backtracing and accuracy measure)
    :return: None.
    """
    parse_command_line_arguments()

    if config.corpus == "brown":  # Default corpus.
        # Only need to do this once.
        # download_brown_corpus()
        # Retrieve tagged sentences from the Brown corpus
        tagged_sentences = brown.tagged_sents(tagset='universal')
        print("Corpus used: Brown Corpus (universal tagset)")
        if config.debug:
            print_corpus_information(brown, "Brown Corpus")
    elif config.corpus == "floresta":
        # Only need to do this once.
        # download_floresta_corpus()
        tagged_sentences = floresta.tagged_sents()
        print("Corpus used: Floresta Treebank")
        if config.debug:
            print_corpus_information(brown, "Floresta Treebank")

    # Start measuring runtime.
    start_time = time.time()

    # Split data into a training and a testing set (default split 95/5 sentences).
    training_set, testing_set = split_train_test_data(tagged_sentences)
    if config.debug:
        print_number_of_sentences(training_set, "training dataset")
        print_number_of_sentences(testing_set, "testing dataset")

    # Replace infrequent words with special 'UNK' tags.
    # training_words = extract_words(training_set)
    # unique_training_words = remove_list_duplicates(training_words)
    # training_set = handle_unknown_words(training_set, unique_training_words, is_training_set=True)
    # testing_set = handle_unknown_words(testing_set, unique_training_words, is_training_set=False)

    # Store all words and all tags from the training dataset in a ordered lists (and make lists without duplicates).
    training_tags = extract_tags(training_set)
    unique_training_tags = remove_list_duplicates(training_tags)

    # Train the POS tagger by generating the tag transition and word emission probability matrices of the HMM.
    tag_transition_probabilities, emission_probabilities = train_tagger(training_set, training_tags)

    # Test the POS tagger on the testing data using the Viterbi back-tracing algorithm.
    test_tagger(testing_set, unique_training_tags, tag_transition_probabilities, emission_probabilities)

    print_runtime(round(time.time() - start_time, 2))  # Record and print runtime.


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their state in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus",
                        default="brown",  # Default corpus is Brown
                        help="The corpus to use."
                        )
    parser.add_argument("-r", "--recalculate",
                        action="store_true",
                        help="Include this flag to recalculate the HMM's tag transition and word emission probability "
                             "matrices. Otherwise, previously trained versions will be used.")
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Include this flag additional print statements and data for debugging purposes.")
    args = parser.parse_args()
    config.corpus = args.corpus
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


def handle_unknown_words(dataset: list, unique_training_words: list, is_training_set: bool = True) -> list:
    """
    Checks for infrequent words and replaces them with appropriate "UNK-x" strings. For the testing set, checks that the
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
    Replaces hapax legomenon (words that appear only once in the data) with a "UNK-x" tag.
    :param dataset: the training dataset.
    :param hapax_legomenon: the list of hapax legomenon.
    :return: the updated dataset.
    """
    for i, sentence in enumerate(dataset):
        for j in range(0, len(sentence)):
            if sentence[j][0] in hapax_legomenon:
                dataset[i][j] = (unknown_words_rules(sentence[j][0]), dataset[i][j][1])
    return dataset


def replace_testing_words(dataset: list, unique_training_words: list) -> list:
    """
    Replaces words from the testing set that do not occur in the training set.
    :param dataset: the testing dataset.
    :param unique_training_words: a list of words that occur in the training set (without duplicates).
    :return: the updated dataset.
    """
    for i, sentence in enumerate(dataset):
        for j in range(0, len(sentence)):
            if sentence[j][0] not in unique_training_words:
                dataset[i][j] = (unknown_words_rules(sentence[j][0]), dataset[i][j][1])
    return dataset


def unknown_words_rules(word: str) -> str:
    """
    Selects a rule to replace a word based on its spelling.
    :param word: the word to parse.
    :return: the new "UNK-x" tag for the word.
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
    Calculates the HMM's smoothed tag transition and word emission probabilities. Checks if previously calculated
    versions exists (loads that into memory using Pickle, otherwise re-calculates the probabilities and saves them to a
    Pickle file).
    :param training_tags: the tags in the training set.
    :param training_set: the training set tokens.
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
        tag_transition_probabilities = get_tag_transition_probabilities(training_set, training_tags)
        # transition_occurrences = count_tag_transition_occurrences(training_tags)
        # tag_transition_probabilities = get_tag_transition_probability(transition_occurrences)
        with open(transition_probabilities_file_path, 'wb') as f:
            pickle.dump(tag_transition_probabilities, f)
        print("File '{0}' regenerated and saved at {0}.".format(transition_probabilities_file_path))

    # Count number of word emissions and get probabilities of emissions.
    # If word emission probabilities were already calculated, then load them into memory, else calculate them.
    emission_probabilities_file_path = "data_objects/emission_probabilities.pkl"
    if os.path.isfile(emission_probabilities_file_path) and not config.recalculate:
        with open(emission_probabilities_file_path, 'rb') as f:
            emission_probabilities = pickle.load(f)
        print("File '{}' already exists, loaded from memory.".format(emission_probabilities_file_path))
    else:
        emission_probabilities = get_word_emission_probabilities(training_set, training_tags)
        with open(emission_probabilities_file_path, 'wb') as f:
            pickle.dump(emission_probabilities, f)
        print("File '{0}' regenerated and saved at {0}.".format(emission_probabilities_file_path))

    # Return the HMM's tag transition and word emission probability matrices.
    return tag_transition_probabilities, emission_probabilities


def get_tag_transition_probabilities(training_set: list, training_tags: list, bins: int = 1000) -> dict:
    """
    Calculates the tag transition probabilities (between a current POS tag and the next POS tag) by smoothing their
    frequency distribution using the Witten-Bell estimate Probability Distribution.
    :param training_tags: the tags in the training set.
    :param training_set: the training set tokens.
    :param bins: the number of bins to use when smoothing.
    :return: the smoothed tag transition probabilities (probability of a POS tag following another POS tag). # todo
    """
    transition_probabilities = dict()
    tags = remove_list_duplicates(training_tags)

    for tag in tags:
        next_tags = list()
        for sentence in training_set:
            for i in range(0, len(sentence) - 1):
                if tag == sentence[i][1]:
                    next_tags.append(sentence[i + 1][1])
        transition_probabilities[tag] = WittenBellProbDist(FreqDist(next_tags), bins=bins)

    return transition_probabilities


def get_word_emission_probabilities(training_set: list, training_tags: list, bins: int = 100000000) -> dict:
    """
    Calculates the word emission probabilities (between the current POS tag and the current word) by smoothing their
    frequency distribution using the Witten-Bell estimate Probability Distribution.
    :param training_tags: the tags in the training set.
    :param training_set: the training set tokens.
    :param bins: the number of bins to use when smoothing.
    :return: the smoothed word emission probabilities (probability of a word given its POS tag). # todo
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
    """
    Execution flow to test the POS tagger using the previously calculated tag transition and word emission
    probabilities. Loops through each sentence in the test set to calculate its viterbi trellis before using a
    back-tracing algorithm to predict the most likely sequence of POS tags. The accuracy is calculated after all
    sentences in the testing dataset have been processed by comparing the predicted tags with the actual tags from
    the selected corpus.
    :param testing_set: the training set sentences.
    :param unique_training_tags: the list tags in the training set without duplicates.
    :param tag_transition_probabilities:
    :param emission_probabilities:
    :return:
    """
    predicted_tags = list()
    # Loop through sentences rather than words to avoid increasingly small probabilities that
    # eventually reach 0 (smallest float value is e-308 in Python).
    for sentence in testing_set:
        testing_words = [w for (w, _) in sentence]
        # Calculate the Viterbi matrix.
        viterbi_trellis = viterbi_algorithm(
            testing_words,
            unique_training_tags,
            tag_transition_probabilities,
            emission_probabilities
        )
        # Use back-tracing to determine the most likely POS tags for each word in the testing dataset.
        predicted_tags.append(backtrace(viterbi_trellis))

    # Print the accuracy of the tagger.
    tagging_accuracy = round(calculate_accuracy(predicted_tags, testing_set), 2)
    print("\nPOS Tagging accuracy on test dataset: {}%".format(tagging_accuracy))


def viterbi_algorithm(words: list, unique_training_tags: list, tag_transition_probabilities: dict,
                      emission_probabilities: dict) -> dict:
    """
    Calculates the viterbi trellis by recording the maximum probability of all POS tags generating each word in the
    testing untagged sentence (only words, no associated POS).
    :param words: a sentence from the testing set.
    :param unique_training_tags:
    :param tag_transition_probabilities: tag transition probabilities (between a current POS tag and the next POS tag).
    :param emission_probabilities: word emission probabilities (between the current POS tag and the current word)
    :return: the Viterbi trellis.
    """
    # Initialise the Viterbi matrix.
    viterbi_trellis = dict()
    for tag in unique_training_tags:
        viterbi_trellis[tag] = {
            "sentence": words,
            "viterbi_value": [0] * len(words)
        }

    # Loop through each word in the testing set.
    for i in range(0, len(words)):
        # Special case for first word, (bigram HMMs, don't care about start of sentence tag <s> before the first word.)
        if i == 0:
            continue
        # Special case: previous tag is <s>.
        elif i == 1:
            for tag in viterbi_trellis.keys():
                viterbi_trellis[tag]["viterbi_value"][i] = \
                    tag_transition_probabilities[config.START_TAG_STRING].prob(tag) * \
                    emission_probabilities[tag].prob(viterbi_trellis[tag]["sentence"][i])
        # All other cases in the sentence.
        else:
            for tag in viterbi_trellis.keys():
                cur_tag_viterbi_values = list()
                for key in viterbi_trellis.keys():
                    previous_viterbi_value = viterbi_trellis[key]["viterbi_value"][i - 1]
                    cur_viterbi_value = previous_viterbi_value * tag_transition_probabilities[key].prob(tag)
                    cur_tag_viterbi_values.append(cur_viterbi_value)
                max_viterbi = max(cur_tag_viterbi_values)
                viterbi_trellis[tag]["viterbi_value"][i] = \
                    max_viterbi * emission_probabilities[tag].prob(viterbi_trellis[tag]["sentence"][i])

        # Unknown word: never seen in the training set. Naive handling of unknown words: set the max_val to 1/1000
        # if viterbi_trellis[tag]["sentence"][i] not in unique_training_words:
        #     for tag in viterbi_trellis.keys():
        #         viterbi_trellis[tag]["viterbi"][i] = \
        #             0.001 * emission_probabilities[viterbi_trellis[tag]["sentence"][i]][tag]

    return viterbi_trellis


def backtrace(viterbi_trellis: dict) -> list:
    """
    Starting from the end of the sentence, select the most likely (highest probability) path of tags leading from the
    end of the sentence to the beginning.
    :param viterbi_trellis: the viterbi trellis generated for the sentence being tagged.
    :return: the (ordered) sequence of most likely POS tags for the given test sentence.
    """
    backwards_predicted_tags = list()
    # Start at end of all sentences and back-trace towards beginning of sentence.
    for i in range(len(viterbi_trellis["."]["sentence"]) - 2, 0, -1):
        cur_predicted_tag = None
        max_viterbi_value = 0
        for tag in viterbi_trellis.keys():
            cur_viterbi_value = viterbi_trellis[tag]["viterbi_value"][i]
            if cur_viterbi_value > max_viterbi_value:
                max_viterbi_value = cur_viterbi_value
                cur_predicted_tag = tag
        backwards_predicted_tags.append((viterbi_trellis[cur_predicted_tag]["sentence"][i], cur_predicted_tag))

    # Reverse the predicted tags to get them back in their original order.
    return reverse_list(backwards_predicted_tags)


def calculate_accuracy(predicted: list, actual: list) -> float:
    """
    Compare the tags predicted from the Viterbi back-tracing algorithm to the tagged version of the testing set.
    :param predicted: the POS tags predicted by the tagger.
    :param actual: the actual sequence of tags extracted from the tagged corpus.
    :return: the accuracy in percentage.
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


if __name__ == "__main__":
    main()
