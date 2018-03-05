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


FILES_LARGE_CORPUS = 300

x, y = loader.load("data/test")
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Entire test corpus", len(x), bleu, accuracy)

x, y = loader.load("data/train", FILES_LARGE_CORPUS)
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Training corpus - {} files".format(FILES_LARGE_CORPUS), len(x), bleu, accuracy)
