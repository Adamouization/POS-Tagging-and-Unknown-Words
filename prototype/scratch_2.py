import nltk
import matplotlib.pyplot as plt
from nltk.corpus import brown
import numpy as np
from collections import Counter
import pandas as pd
from itertools import chain

def get_train_test(sents):
    training = list(sents[:train_size])
    test = list(sents[train_size:train_size + test_size])

    ##append start and end tags to training data
    for s in training:
        s = (s.insert(0, START_TAG_TUPLE))
    for s in training:
        s = (s.insert(len(s), END_TAG_TUPLE))

    ##append start and end tags to training data
    for s in test:
        s = (s.insert(0, START_TAG_TUPLE))
    for s in test:
        s = (s.insert(len(s), END_TAG_TUPLE))

    training_words = list(chain.from_iterable(training))
    test_words = list(chain.from_iterable(test))

    return training_words, test_words


def get_transition_counts(tags: list) -> dict:
    """
    For each tag in the list of tags, fnd the probability distribution of the next tags that follow

    :param tags:
    :return:
    """

    word_list = []
    # words = brown.tagged_words(tagset="universal")

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

def get_emission_counts(words: list, tags: list) -> dict:
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

def get_transition_proba(transitions):
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

def get_emission_proba(emissions):
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



sents = brown.tagged_sents(tagset='universal')
train_size = 10000
test_size = 500
START_TAG_TUPLE = ("<s>", "<s>")
END_TAG_TUPLE = ("</s>", "</s>")
START_TAG = ("<s>")
START_TAG = ("</s>")

train, test = get_train_test(sents)
sents_in_train = sum(x == "<s>" for (x, _) in train)
sents_in_test = sum(x == "<s>" for (x, _) in test)
print("Num of sentencs in training is: {}".format(sents_in_train))
print("Num of sentencs in test is    : {}".format(sents_in_test))

words = [w for (w, _) in train]
tags = [p for (_, p) in train]

transition_counts = get_transition_counts(tags)
transition_proba = get_transition_proba(transition_counts)

emissions = get_emission_counts(words, tags)
emission_proba = get_emission_proba(emissions)

## get the list of words from test list
seq = [w for (w, _) in test[:52]]
unique_words = set(seq)
unique_tags = set(tags)
preicted_tags = []
num_sentences = sum(x == "<s>" for x in seq)
viterbi_proba = []

##createes the viterbi matrix/table shit
viterbi_matrix = {}
for tag in unique_tags:
    viterbi_matrix[tag] = {}

    val = {}
    val["words"] = seq
    val["viterbi"] = [0] * len(seq)

    viterbi_matrix[tag] = val


## Compute the viterbi probabilities for each word
for i in range(0, len(seq)):
    if viterbi_matrix[tag]["words"][i] in unique_words:
        if i == 0:
            continue

        elif i == 1:
            for tag in viterbi_matrix.keys():
                current_word = viterbi_matrix[tag]["words"][i]

                viterbi_matrix[tag]["viterbi"][i] = transition_proba["<s>"][tag] * emission_proba[current_word][tag]
                previous_word = viterbi_matrix[tag]["words"][i]

        else:
            for tag in viterbi_matrix.keys():
                temp = []
                for k in viterbi_matrix.keys():
                    previous_viterbi = viterbi_matrix[k]["viterbi"][i - 1]

                    val = previous_viterbi * transition_proba[k][tag]
                    temp.append(val)
                max_val = max(temp)

                current_word = viterbi_matrix[tag]["words"][i]

                viterbi_matrix[tag]["viterbi"][i] = max_val * emission_proba[current_word][tag]

    ## word not in unique words:
    ## set viterbis to 1/1000
    else:
        for tag in viterbi_matrix.keys():
            viterbi_matrix[tag]["viterbi"][i] = 0.001 * emission_proba[current_word][tag]