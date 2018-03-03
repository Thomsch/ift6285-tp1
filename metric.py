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
    Computes the word to word ratio between references and translation of correct translations.
    If there is more or less words in the translation, they are counted as false.
    :param references: The list of reference sentences as strings.
    :param translations: The list of translated sentences as strings.
    :return: The percentage of correctly translated word on the total number of words in the reference sentence.
    """
    test_acc_count, count_total = count_accuracy(references, translations)
    return test_acc_count/count_total


def count_accuracy(pred_sents,target_sents):
    count_accu=0
    total=0
    for pred_sent,target_sent in zip(pred_sents,target_sents):
        print(pred_sent)
        print(target_sent)
        pred_list=pred_sent.split(' ')
        targ_list=target_sent[2:].split(' ')
        for pred_token,target_token in zip(pred_list,targ_list):
            total+=1
            if pred_token==target_token:
                count_accu+=1
    return count_accu, total
