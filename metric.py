import nltk.translate.bleu_score as nltk

WEIGHTS = (1, 0)


def bleu_sentence(reference, translation):
    """
    Wrapper for blue_sentence from NLTK.
    :param reference: The reference sentence as a string.
    :param translation: the translation sentence as a string.
    :return: The BLEU score for the sentence
    """
    return nltk.sentence_bleu([reference.split()], translation.split(), WEIGHTS)


def bleu_corpus(references, translations):
    """
    Wrapper for bleu_corpus from NLTK.
    :param references: The list of reference sentences as strings.
    :param translations: The list of translated sentences as strings.
    :return: The BLEU score for the corpus
    """
    references = [[sentence.split()] for sentence in references]
    translations = [sentence.split() for sentence in translations]
    return nltk.corpus_bleu(references, translations, WEIGHTS)
