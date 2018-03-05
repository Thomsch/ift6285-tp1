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


TEST_CORPUS_SIZE = 20

x, y = loader.load("data/test", TEST_CORPUS_SIZE)
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Entire test corpus", len(x), bleu, accuracy)

x, y = loader.load("data/train", 500)
bleu = metric.bleu_corpus(y, x)
accuracy = metric.accuracy(y, x)
display_result("Training corpus - 500 files", len(x), bleu, accuracy)

bleu = metric.bleu_corpus(y[:TEST_CORPUS_SIZE], x[:TEST_CORPUS_SIZE])
accuracy = metric.accuracy(y, x)
display_result("Training corpus - match test corpus", TEST_CORPUS_SIZE, bleu, accuracy)
