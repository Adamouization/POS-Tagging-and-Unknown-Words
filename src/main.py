from nltk.corpus import brown

from src.helpers import download_brown_corpus, print_corpus_information


def main():
    counting_occ_pos = dict()
    for sentence in brown.tagged_sents(tagset='universal'):
        tags = [t for (_, t) in sentence]
        current_tag, previous_tag = (str(),) * 2

        for i, tag in enumerate(tags):
            # Special case: first tagged token does not have any preceding token.
            if i == 0:
                current_tag = tag
            else:
                previous_tag = current_tag
                current_tag = tag

                # Create a new nested current tag dict in the dictionary.
                if current_tag not in counting_occ_pos.keys():
                    counting_occ_pos[current_tag] = dict()

                # Create a new preceding tag value in the nested dict.
                if previous_tag not in counting_occ_pos[current_tag].keys():
                    counting_occ_pos[current_tag][previous_tag] = 0

                # Increment the occurence once we are sure that the keys are correctly created in the dict.
                counting_occ_pos[current_tag][previous_tag] += 1


if __name__ == "__main__":
    main()
