"""
This script computes the baseline results. The output of these results can be found in the report.
"""

import loader
import metric


def display_result(test_name, sentences_number, bleu_score, accuracy):
    print(test_name)
    print("=========================")
    print("# sentence: {}".format(sentences_number))
    print("BLEU score: {}".format(bleu_score))
    print("Exactitude: {}".format(accuracy))
    print()


x, y = loader.load("data/test")
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Entire test corpus", len(x), bleu, accuracy)

x, y = loader.load("data/train", 500)
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Training corpus - 500 files", len(x), bleu, accuracy)
