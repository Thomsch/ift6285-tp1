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


def accuracy(references, translations):
    """
    Computes the average accuracy per sentence.
    :param references: The list of reference sentences as strings.
    :param translations: The list of translated sentences as strings.
    :return: The average accuracy per sentence.
    :raise ValueError: If ``references`` and ``translations`` do not have the same length.
    """
    if len(references) != len(translations):
        raise ValueError("Lists must have the same length.")

    accuracies = []
    for reference, translation in zip(references, translations):
        accuracies.append(sentence_accuracy(reference, translation))
    return sum(accuracies) / len(accuracies)


def sentence_accuracy(reference_sentence, translated_sentence):
    """
    Computes the ratio of correctly translated words in a reference sentence.
    If the translation contains more or less words than the reference, they are counted as ``false``.
    :param reference_sentence: The reference sentence as a string
    :param translated_sentence: The translation sentence as a string
    :return: the accuracy for a sentence.
    """
    correct = 0
    total = 0

    references = reference_sentence.split()
    translations = translated_sentence.split()
    for reference, translation in zip(references, translations):
        if reference == translation:
            correct += 1
        total += 1

    total += abs(len(references) - len(translations))
    return correct / total


def display_result(test_name, sentences_number, accuracy_score, bleu_score):
    print(test_name)
    print("=========================")
    print("# sentence: {}".format(sentences_number))
    print("Exactitude: {}".format(accuracy_score))
    print("BLEU score: {}".format(bleu_score))
    print()
