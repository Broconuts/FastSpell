import re
import os
import csv
import collections
import enchant
import nltk
import pandas as pd
from gensim.models.fasttext import FastText, load_facebook_vectors


class FastSpell:
    """
    Error detection and spelling correction using FastText word embeddings to generate a list of errors and their
    correct counterparts in a given corpus.
    """

    def __init__(self, path, pretrained=False):
        self.frequency_list = self.build_frequency_list(path)
        # self.frequency_stats()
        # Todo: implement a load_frequency_list function as an option
        if pretrained:
            self.embeddings = self.load_embeddings(path)
        else:
            # Todo: implement transfer-learning
            self.embeddings = self.build_embeddings(path, overwrite=False)
        # Todo: implement multi-language support
        self.dictionary_us = enchant.Dict('en-US')
        self.dictionary_gb = enchant.Dict('en-GB')
        self.recognize_mistakes()

    def build_embeddings(self, corpus_path, model_path='models', transfer=False, overwrite=False):
        """
        Build FastText embeddings for a given corpus. Please note that FastText needs to be installed in the active
        environment for this function to work (but it is not technically in the list of requirements).
        :param corpus_path: The path to the text corpus used to generate embeddings.
        :type corpus_path: str
        :param model_path: The path where the word embeddings are to be stored.
        :type model_path: str
        :param transfer: Encodes whether the new embeddings should be added "on top" of existing embeddings.
        :type transfer: bool
        :param overwrite: If a trained model already exists but the user still wants to train one from scratch, this is
        true.
        :type overwrite: bool
        """
        # if there already exists a model in the given directory, load it instead of training a new one
        if not overwrite and len(os.listdir(model_path)) != 0:
            return self.load_embeddings(model_path + "/fasttext.model")
        model = FastText()
        # build the vocabulary
        model.build_vocab(corpus_file=corpus_path)
        # train the model
        model.train(corpus_file=corpus_path, epochs=5,
                    total_examples=model.corpus_count, total_words=model.corpus_total_words)

        model.save(model_path + "/fasttext.model")
        return model

    def load_embeddings(self, path):
        """
        Loads existing embeddings to use for spelling correction.
        :param path: Path to the embeddings.
        :type path: str
        """
        return FastText.load(path)

    def build_frequency_list(self, path):
        """
        Build frequency list for a given corpus.
        :param path: The path to the text corpus used to generate frequency list.
        :type path: str
        :returns A frequency list for a given text corpus.
        """
        # set of characters we want to remove from our strings
        chars_to_remove = r"['\",.?!()/]"
        # tokenize sentences, store individual words in list
        words = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Todo: evaluate whether reading in strings as all-lowers might cause problems
                line = re.sub(chars_to_remove, ' ', line.strip().lower())
                words.extend(line.split())
        # return frequency list from tokenized word list
        return collections.Counter(words)

    def frequency_stats(self):
        """
        Provides statistics on the corpus. Useful to assess how well this approach to spelling correction might work
        on the given corpus.
        """
        print("Total number of words: %s" % len(self.frequency_list))
        counter = pd.Series([self.frequency_list[word] for word in self.frequency_list])
        print(counter.value_counts())

    def recognize_mistakes(self, similarity_threshold=0.96, frequency_threshold=10, character_minimum=3,
                           identical_starting_letter=True) -> dict:
        """
        Check if a given word passes the criteria on whether or it constitutes a mistake.

            similarity_threshold: If the similarity between the vector for the less frequent word (i.e. the mistake
                                     candidate) and the one for the more frequent word is lower than this threshold,
                                     do not consider the words related and dismiss mistake candidate.
            frequency_threshold: The minimum amount of occurrences of a word we look for "mistakes" of for.
            character_minimum: A mistake candidate must exceed this number of characters. A character minimum is
                                  useful in preserving domain-specific abbreviations and acronyms as non-errors.
            identical_starting_letter: Typos usually don't occur with the first letter of a word. To limit
                                          false positives in mistake detection, this condition can either be taken into
                                          consideration (variable = True) or dismissed (variable = False).
        """
        # use defaultdict rather than python's default dict for easily adding new keys
        error_dict = collections.defaultdict(list)
        # go through frequency list to find neighbors to the most frequent (therefore likely correct) words
        for word, frequency in self.frequency_list.most_common():
            # only consider words that occur at least n times in the corpus to be "correct" baselines
            if frequency > frequency_threshold:
                # Todo: tweak topn based on the average number of most similar word vectors above the
                #  similarity_threshold
                # generate list of nearest neighbors
                candidates = self.embeddings.wv.most_similar(word, topn=25)
                for word_candidate, score in candidates:
                    # if candidate has lower similarity score than specified threshold or is shorter than the specified
                    # character minimum, we can stop checking
                    if score < similarity_threshold or len(word_candidate) < character_minimum: break
                    # if we specified the identical-starting-letter criterion, skip this word candidate if it does not
                    # fulfill it
                    if identical_starting_letter and word[0] != word_candidate[0]: continue
                    # if word candidate can be found in a dictionary, do not consider it a typo
                    # TODO: add check that the word is not a plural (i.e. added "s" in the end) of known word
                    # TODO: deal with stuff like "centricity"
                    if self.dictionary_gb.check(word_candidate) or self.dictionary_us.check(word_candidate): continue
                    # if Levenshtein distance of word candidate to original word is larger than 2 for words longer than
                    # four characters or 1 for words with fewer, do not consider them related and therefore not a typo
                    # of the original word
                    if nltk.edit_distance(word, word_candidate) <= (1 if len(word_candidate) <= 4 else 2): continue
                    # if error occurs only twice or fewer times, do not consider it an error
                    # removed due to limited size of dataset
                    # if self.frequency_list[word_candidate] >= 2: continue
                    # if the candidate matches all the criteria for being a typo, add typo-word pair to the error dict
                    error_dict[word_candidate].append(word)
        print(error_dict)
        return error_dict


if __name__ == '__main__':
    fs = FastSpell("data/corpus.txt")
