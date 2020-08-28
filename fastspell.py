import re
import os
import csv
import collections
import time
import enchant
import nltk
import pandas as pd
from tqdm import tqdm

from gensim.models.fasttext import FastText, load_facebook_vectors, load_facebook_model


class FastSpell:
    """
    Error detection and spelling correction using FastText word embeddings to generate a list of errors and their 
    correct counterparts in a given corpus.
    
    Attributes:
        corpus:         A list of sentences that is to be corrected.
        frequency_list: A dict containing all unique words that occurr in the corpus, as well as their frequencies.
        embeddings:     FastText word embeddings based on the corpus.
        dictionary_us:  A dictionary for words in US-English.
        dictionary_gb:  A dictionary for words in Oxford-English.
        error_dict:     A dictionary containing (1) words from the corpus that were identified as typos as well as 
                        (2) their suggested corrections.
        
    Functions:
        load_corpus:                    Load the text from a file.
        get_frequency_list:             Generate the frequency list.
        get_embeddings:                 Generate the word embeddings or load existing ones.                    
        recognize_mistakes_in_corpus:   Generate error dictionary.
    """

    def __init__(self, path: str):
        """
        Initiates spellcorrector on given text corpus by generating frequency list, word embeddings, and error dict on
        the given corpus.
        
        :param path: The location of the text document containing the text corpus that is to be spell-checked
        """
        self.corpus = self.load_corpus(path)
        
        self.embeddings = self.get_embeddings(path, overwrite=False)

        
        self.frequency_list = self.get_frequency_list()
            
        # TODO: implement multi-language support
        # load dictionaries
        self.dictionary_us = enchant.Dict('en-US')
        self.dictionary_gb = enchant.Dict('en-GB')
        
        # generate error dictionary
        # self.error_dict = self.v2_recognize_mistakes_in_corpus()


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
        print("Loading text corpus.")
        for text in df["answers"]:
            answer = []
            for token in text.split():
                answer.append(self.clean_token(token))
            corpus.append(answer)

        return corpus


    def clean_token(self, token: str) -> str:
        """
        Cleans a given token by removing special characters and making them lowercase.

        :param token: The token that is to be cleaned.
        :returns: A cleaned version of the given token.
        """
        clean_token = ""
        for i, char in enumerate(list(token)):
            if re.match(r"\W", char):
                # if character is not one that might carry meaning for the word, do not add it to the clean token
                if char not in ["'", "-"]:
                    continue
                # if apostrophe is not in a valid position (i.e. the second to last token in the word), do not add it
                elif char == "'" and i != len(list(token)) - 2:
                    continue
            clean_token += char.lower()
        return clean_token


    def get_frequency_list(self) -> collections.Counter:
        """
        Generate a dictionary containing the frequency of every word in our newly trained word embeddings.

        :returns: A frequency list for the word embeddings.
        """
        # clean data and count individual words
        frequency_list = collections.Counter()
        
        # TODO: use identical tokenization process to the one used in pipeline
        print("Generate frequency list.")
        for answer in self.corpus:
            frequency_list.update(answer)
            
        # return frequency list from tokenized word list
        return frequency_list
        

    def get_embeddings(self, corpus_path: str, model_path: str = 'models', transfer: bool = False, 
                        overwrite: bool = False):
        """
        Build FastText embeddings for a given corpus if no embeddings exist yet or existing embeddings are to be 
        overwritten. Loads and returns existing embeddings if they can be detected. TODO: implement that last bit.

        :param corpus_path: The path to the text corpus used to generate embeddings.
        :param model_path: The path where the word embeddings are to be stored.
        :param transfer: Encodes whether the new embeddings should be added "on top" of existing embeddings.
        :param overwrite: If a trained model already exists but the user still wants to train one from scratch, this is 
                          true.
        """
        if overwrite or len(os.listdir("models/transfer_learned")) == 0:
            print("Loading pretrained model...")
            model = load_facebook_model("models\wiki.en.bin")
            model.build_vocab(sentences=self.corpus, update=True)
            print("Successfully loaded pretrained model!\nStart transfer-learning...")
            model.train(sentences=self.corpus, total_examples=len(self.corpus), epochs=5)
            print("Successfully finished transfer learning!")
            model.save("models/transfer_learned/big_model.model")
        else:
            print("Loading word embeddings...")
            model = FastText.load("models/transfer_learned/big_model.model")
            print("Word embeddings loaded!")

        return model

    
    def is_error(self, word: str, frequency_threshold: int = 10, character_minimum: int = 3) -> bool:
        """
        Check if a given word passes the criteria on whether or it constitutes a mistake.

        :param word: The word that is to be checked for being an error
        :param frequency_threshold: The minimum amount of occurrences of a word we look for "mistakes" of for.
        :param character_minimum: A mistake candidate must exceed this number of characters. A character minimum is 
                                  useful in preserving domain-specific abbreviations and acronyms as non-errors.
        """
        # clean token to ensure uniformity with the frequency list
        word = self.clean_token(word)
        # if error candidate appears too often in the corpus, do not consider it further as an error
        if self.frequency_list[word] > frequency_threshold:
            return False
        # if word is too short, don't consider it an error candidate
        if len(word) < character_minimum:
            return False            
        # if word candidate can be found in a dictionary, do not consider it a typo
        # TODO: add check that the word is not a plural (i.e. added "s" in the end) of known word
        # TODO: deal with stuff like "centricity" where only "centric" is in the dict and "limitlessly" 
        #  where only "limitless" is in the dict
        if (self.dictionary_gb.check(word) or self.dictionary_us.check(word)):
            False
        
        # if none of the criteria are met, consider the word an error
        return True


    def find_correction_suggestion(self, error_word: str, similarity_threshold: float = 0.96, 
                                    identical_starting_letter: bool = True) -> list:
        """
        Attempts to find a correct suggestion for an incorrect word.

        :param error_word: The word for which we need to find a correction.
        :param similarity_threshold: The minimum similarity between the error candidate and the correction candidate for
                                     them to be considered related.
        :param identical_starting_letter: Typos usually don't occur with the first letter of a word. To limit false
                                          positives in mistake detection, this condition can either be taken into
                                          consideration (variable = True) or dismissed (variable = False).
        :returns: A list of strings that could ne the correct version of the word the input string was supposed to be.
        """
        candidates = self.embeddings.wv.most_similar(error_word, topn=25)
        suggestions = []
        for word_candidate, score in candidates:
            # if candidate has lower similarity score than specified threshold or is shorter than the 
            #  specified character minimum, skip candidate
            if score < similarity_threshold:
                continue
            # if we specified the identical-starting-letter criterion, skip this word candidate if it does 
            #  not fulfill it
            if identical_starting_letter and word[0] != word_candidate[0]:
                continue
            # if Levenshtein distance of word candidate to original word is larger than 2 for words longer 
            #  than four characters or 1 for words with fewer, do not consider them related as they are not
            #  a typo of the original word
            if nltk.edit_distance(word, word_candidate) <= (1 if len(word_candidate) <= 4 else 2): 
                continue
            # if the candidate matches all the criteria for being a typo, add typo-word pair to the error 
            #  dict                
            suggestions.append(word_candidate)

        return suggestions

    def check_and_correct(self, word: str, verbose: bool = False) -> str:
        """
        Checks if a given word is an error and - if so - replaces it with the most likely suggested correction. If the word
        is considered an error but no suitable correction can be found, return the original word.

        :param word: The word that is to be checked and corrected.
        :param verbose: If true, provide additional information about the correction process on the console.
        :returns: A string that is either the original word or a corrected version of the word.
        """
        if self.is_error(word):
            if verbose:
                print(f"Spelling error found: {word}")
            suggested_spellings = self.find_correction_suggestion(word)
            if verbose:
                print("Suggestions for correction:")
                print(suggested_spellings)
            if suggested_spellings:
                return suggested_spellings[0]
        return word


if __name__ == '__main__':
    """Immediate testing."""
    start_time = time.time()
    fs = FastSpell("data/pnlp_data.csv")
    print("FastSpell object generated after", str(time.time() - start_time), "seconds.")
    for answer in fs.corpus:
        for word in answer:
            fs.check_and_correct(word, verbose=True)
