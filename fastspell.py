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
    Error detection and spelling correction using FastText word embeddings to 
    generate a list of errors and their correct counterparts in a given corpus.
    
    Attributes:
        corpus:
        frequency_list:
        embeddings:
        dictionary_us:
        dictionary_gb:
        
    Functions:
        get_embeddings:
        get_frequency_list:
        load_corpus:
        recognize_mistakes_in_corpus:
    """

    def __init__(self, path, pretrained: bool = False):
        """
        Initiates FastSpell with a location and a model?
        
        :param path: ehm? a path to something?
        :param pretrained: to use a pretrained embedding?
        """
        self.corpus = self.load_corpus(path)
        self.frequency_list = self.get_frequency_list()
        
        # TODO: implement a load_frequency_list function as an option
        if pretrained:
            self.embeddings = self.get_embeddings(path)
        else:
            # TODO: implement transfer-learning
            self.embeddings = self.get_embeddings(path, overwrite=False)
            
        # TODO: implement multi-language support
        # load dictionaries
        self.dictionary_us = enchant.Dict('en-US')
        self.dictionary_gb = enchant.Dict('en-GB')
        
        # generate error dictionary
        self.recognize_mistakes_in_corpus()


    def load_corpus(self, path: str) -> list:
        """
        Loads the text that is to be corrected and tokenizes it.

        :param path: The location of the file containing the text.
        :returns: A list of lists containing answers tokenized on word-level.
        """
        # TODO: add error handling for file-loading
        # load in data
        df = pd.read_csv(path, encoding="utf-8")
        corpus = []
        
        # TODO: use identical tokenization process to the one used in pipeline
        # tokenize each answer into a list of words
        for text in df["answers"]:
            answer = []
            for word in text.split():
                answer.append(word)
            corpus.append(answer)
            
        return corpus


    def get_frequency_list(self) -> collections.Counter:
        """
        Generate a dictionary containing the frequency of every word in the 
        given corpus.

        :returns: A frequency list for a given text corpus.
        """
        # clean data and count individual words
        frequency_list = collections.Counter()
        frequency_list.clear()
        
        # TODO: use identical tokenization process to the one used in pipeline
        for answer in self.corpus:
            frequency_list.update(answer)
            
        # return frequency list from tokenized word list
        return frequency_list
        

    def get_embeddings(self, corpus_path: str, model_path: str = 'models', 
        transfer: bool = False, overwrite: bool = False):
        """
        Build FastText embeddings for a given corpus if no embeddings exist yet
        or existing embeddings are to be overwritten. Loads and returns existing 
        embeddings if they can be detected. TODO: implement that last bit.

        :param corpus_path: The path to the text corpus used to generate 
                            embeddings.
        :param model_path: The path where the word embeddings are to be stored.
        :param transfer: Encodes whether the new embeddings should be added "on
                         top" of existing embeddings.
        :param overwrite: If a trained model already exists but the user still
                          wants to train one from scratch, this is true.
        """
        # check if embeddings already exist to skip redundant training
        overwrite = False
        
        if not overwrite and len(os.listdir("models")) != 0:
            model = FastText.load("models/fasttext.model")
        else:
            model = FastText()
            # build the vocabulary
            model.build_vocab(sentences=self.corpus)
            # train the model
            model.train(self.corpus, epochs=5, sg=1,
                        total_examples=model.corpus_count, 
                        total_words=model.corpus_total_words)

            model.save("models/fasttext.model")
            
        return model


    def recognize_mistakes_in_corpus(self, similarity_threshold: float = 0.96, 
        frequency_threshold: int = 10, character_minimum: int = 3,
        identical_starting_letter: bool = True) -> dict:
        """
        Check if a given word passes the criteria on whether or it constitutes
        a mistake.

        :param similarity_threshold: If the similarity between the vector for 
                                     the less frequent word (i.e. the mistake 
                                     candidate) and the one for the more 
                                     frequent word is lower than this threshold,
                                     do not consider the words related and 
                                     dismiss mistake candidate.
        :param frequency_threshold: The minimum amount of occurrences of a word
                                    we look for "mistakes" of for.
        :param character_minimum: A mistake candidate must exceed this number of
                                  characters. A character minimum is useful in 
                                  preserving domain-specific abbreviations and 
                                  acronyms as non-errors.
        :param identical_starting_letter: Typos usually don't occur with the 
                                          first letter of a word. To limit false
                                          positives in mistake detection, this
                                          condition can either be taken into
                                          consideration (variable = True) or
                                          dismissed (variable = False).
        """
        dictionary_gb = enchant.Dict('en-GB')
        dictionary_us = enchant.Dict('en-US')

        # use defaultdict rather than python's default dict for easily adding 
        #  new keys
        error_dict = collections.defaultdict(list)
        missing_words = []
        
        with open("log.csv", "w", encoding="utf-8") as file:
            logger = csv.writer(file)
            logger.writerow(["Correct Word",
                             "Error Candidate",
                             "Criterion that was not met"])
            
            # go through frequency list to find neighbors to the most frequent 
            #  (therefore likely correct) words
            for word, frequency in self.frequency_list.most_common():
                unmatched_criteria = []  # helper variable for logging
                
                # only consider words that occur at least n times in the corpus 
                # to be "correct" baselines
                if frequency > frequency_threshold:
                    # TODO: tweak topn based on the average number of most 
                    # similar word vectors above the similarity_threshold
                    # generate list of nearest neighbors
                    candidates = self.embeddings.wv.most_similar(word, topn=25)
                    
                    for word_candidate, score in candidates:
                        # if candidate has lower similarity score than specified 
                        #  threshold or is shorter than the specified
                        # character minimum, we can stop checking
                        if score < similarity_threshold:
                            logger.writerow([word,
                                             word_candidate,
                                             "similarity_threshold"])
                            break
                        if len(word_candidate) < character_minimum:
                            unmatched_criteria.append("word_length")
                            # continue
                        # if we specified the identical-starting-letter
                        # criterion, skip this word candidate if it does not
                        # fulfill it
                        if (identical_starting_letter and 
                            word[0] != word_candidate[0]): 
                            unmatched_criteria.append("identical_first_letter")
                            # continue
                        # if word candidate can be found in a dictionary, do not 
                        #  consider it a typo
                        # TODO: add check that the word is not a plural (i.e. 
                        # added "s" in the end) of known word
                        # TODO: deal with stuff like "centricity" where only 
                        #  "centric" is in the dict and "limitlessly" where only 
                        #  "limitless is in the dict"
                        if (dictionary_gb.check(word_candidate) or 
                            dictionary_us.check(word_candidate)): 
                            unmatched_criteria.append("dict_check")
                            # continue
                        # if Levenshtein distance of word candidate to original
                        # word is larger than 2 for words longer than four 
                        # characters or 1 for words with fewer, do not consider
                        # them related --> not a typo of the original word
                        if (nltk.edit_distance(word, word_candidate) <= 
                            (1 if len(word_candidate) <= 4 else 2)): 
                            unmatched_criteria.append("edit_distance")
                            # continue
                        # if error occurs only twice or fewer times, do not 
                        # consider it an error
                        # if frequency_list[word_candidate] >= 2: continue
                        # if the candidate matches all the criteria for being a 
                        # typo, add typo-word pair to the error dict
                        # if one or more of the conditions was met, skip word
                        if unmatched_criteria:
                            logger.writerow([word,
                                            word_candidate,
                                            unmatched_criteria])
                            continue
                        
                        error_dict[word_candidate].append(word)
                        
        print(error_dict)
        return error_dict


if __name__ == '__main__':
    """Immediate testing."""
    fs = FastSpell("data/pnlp_data.csv")
