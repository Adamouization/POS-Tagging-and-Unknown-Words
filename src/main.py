from nltk.corpus import brown

from src.helpers import download_brown_corpus, print_corpus_information


def main():
    # tag_transition_occurrences = count_tag_transition_occurrences()
    word_tag_pairs = count_word_tag_pairs()


def count_tag_transition_occurrences():
    tag_transition_occurrences = dict()

    # Loop through each sentence in the Brown corpus.
    for sentence in brown.tagged_sents(tagset='universal'):

        # Collect all tags in the sentence.
        tags = [t for (_, t) in sentence]

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


def count_word_tag_pairs():
    word_tag_pairs_occurrences = dict()

    # Loop through each sentence in the Brown corpus.
    for sentence in brown.tagged_sents(tagset='universal'):

        # Loop through each tagged word in the sentence.
        for tagged_word in sentence:
            word = tagged_word[0]
            tag = tagged_word[1]

            # Create a new nested current tag dict in the dictionary.
            if word not in word_tag_pairs_occurrences.keys():
                word_tag_pairs_occurrences[word] = dict()

            # Create a new preceding tag value in the nested dict.
            if tag not in word_tag_pairs_occurrences[word].keys():
                word_tag_pairs_occurrences[word][tag] = 0

            # Increment the occurrence once we are sure that the keys are correctly created in the dict.
            word_tag_pairs_occurrences[word][tag] += 1

    return word_tag_pairs_occurrences


if __name__ == "__main__":
    main()
