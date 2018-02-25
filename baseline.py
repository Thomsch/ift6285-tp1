"""
This script computes the baseline results. The output of these results can be found in the report.
"""

import loader
import metric


def display_result(test_name, bleu_score, sentences_number):
    print(test_name)
    print("=========================")
    print("# sentence: {}".format(sentences_number))
    print("BLEU score: {}".format(bleu_score))
    print()


TEST_CORPUS_SIZE = 20
x, y = loader.load("data/test", TEST_CORPUS_SIZE)
bleu = metric.bleu_corpus(y, x)
display_result("Entire test corpus", bleu, len(x))

x, y = loader.load("data/train", 500)
bleu = metric.bleu_corpus(y, x)
display_result("Training corpus - 500 files", bleu, len(x))

bleu = metric.bleu_corpus(y[:TEST_CORPUS_SIZE], x[:TEST_CORPUS_SIZE])
display_result("Training corpus - match test corpus", bleu, TEST_CORPUS_SIZE)
