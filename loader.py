import gzip
import glob


def load(folder, number_of_files):
    """
    Loads the sentences contained in the first n zipped files of a folder.
    The final point of the sentence is remove and the sentences are transformed into lowercase.
    :param folder: the folder to load the data from.
    :param number_of_files: the number of files to load.
    :return: a tuple containing (the list of lemmatized sentences, the list of original sentences)
    """
    files = list_files(folder, number_of_files)

    lemmatized_sentences = []
    original_sentences = []
    for zipped_file in files:
        print("Parsing " + zipped_file)
        file = unzip(zipped_file)
        (lemmas, originals) = parse(file)
        original_sentences += originals
        lemmatized_sentences += lemmas

    return lemmatized_sentences, original_sentences


def list_files(folder, number_of_files):
    """
    Returns the path of the first n *.gz files in a folder.
    :param folder: The folder
    :param number_of_files: The number of files to list
    :return: The list of the files' path
    """
    paths = glob.glob(folder + "/*.gz")
    return paths[:number_of_files]


def parse(file):
    """
    Parse a file into a tuple: (list of lemmatized sentences, list of original sentences). All the sentences
    are converted into lower case and final point of the phrase is removed.
    :param file: the file to parse
    :return: the tuple
    """
    lines = file.splitlines()

    lemmatized_sentences = []
    original_sentences = []

    lemmatized_sentence = ""
    original_sentence = ""

    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue

        split = line.split()
        if len(split) != 2:
            continue

        word, lemma = split

        if lemma == '.' and word == '.':
            original_sentences.append(original_sentence.rstrip())
            lemmatized_sentences.append(lemmatized_sentence.rstrip())
            original_sentence = ""
            lemmatized_sentence = ""
        else:
            original_sentence += word.lower() + ' '
            lemmatized_sentence += lemma.lower() + ' '

    return lemmatized_sentences, original_sentences


def unzip(zipped_file):
    """
    Unzip a file and returns it in form of a list of lines.
    :param zipped_file: the path of the zipped file
    """
    with gzip.open(zipped_file, 'rt', encoding='ISO-8859-1') as file:
        file = file.read()
    return file
