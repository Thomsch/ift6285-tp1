import unittest
import metric


class TestMetric(unittest.TestCase):

    def test_bleu_sentence_corpus_should_be_the_same(self):
        reference = "qalaye niazi is an ancient fortified area in paktia province in afghanistan"
        hypothesis = "qalaye niazi be a ancient fortified area in paktia province in afghanistan"
        score_sentence = metric.bleu_sentence(reference, hypothesis)
        score_corpus = metric.bleu_corpus([reference], [hypothesis])

        self.assertEquals(score_corpus, score_sentence)