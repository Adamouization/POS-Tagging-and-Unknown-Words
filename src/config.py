"""
Variables set by the command line arguments dictating which parts of the program to execute, and constants.
"""


# Constants
START_TAG_TUPLE = ("<s>", "<s>")
END_TAG_TUPLE = ("</s>", "</s>")
START_TAG_STRING = "<s>"
END_TAG_STRING = "</s>"
DEFAULT_TRAIN_SIZE = 10000
DEFAULT_TEST_SIZE = 500
MAX_SENTENCE_LENGTH = 150

NOUN_SUFFIX = ["action", "age", "ance", "cy", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
               "ment", "ness", "or", "ry", "scape", "ship", "dom", "ty"]
VERB_SUFFIX = ["ed", "ify", "ise", "ize", "ate", "ing"]
ADJ_SUFFIX = ["ous", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "able"]
ADV_SUFFIX = ["wise", "wards", "ward"]


# Global variables (set by command-line arguments).
corpus = "brown"        # Default corpus to use.
recalculate = False     # Recalculate the HMM's tag transition and word emission probability matrices.
debug = False           # Boolean used to print additional logs for debugging purposes.
