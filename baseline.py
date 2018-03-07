"""
This script computes the baseline results. The output of these results can be found in the report.
"""

import loader
import metric


def display_result(test_name, sentences_number, accuracy, bleu_score):
    print(test_name)
    print("=========================")
    print("# sentence: {}".format(sentences_number))
    print("Exactitude: {}".format(accuracy))
    print("BLEU score: {}".format(bleu_score))
    print()


FILES_LARGE_CORPUS = 300

x, y = loader.load("data/test")
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Test corpus", len(x), accuracy, bleu)

x, y = loader.load("data/train", FILES_LARGE_CORPUS)
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Training corpus - {} files".format(FILES_LARGE_CORPUS), len(x), accuracy, bleu)
