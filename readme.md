# Spelling Correction
## 1. Problem Definition
A. **Error Detection:** Identifying an incorrect word in a sequence of words.

B. **Candidate Generation:** Generate a list of candidates that are likely correct alternatives to the incorrect words.

C. **Candidate Selection:** Select the candidate from the list of candidates that is the most likely correct one.


### 1.1 A Note on "(In)Correctness"
There are different kinds of incorrectness; for the sake of brevity, let us focus on two (and simplify the underlying issues somewhat). An incorrect word can be (1) a typo that is not in the dictionary (e.g. "ther" instead of "there"), or (2) a typo that is still technically a word (e.g. "tree" instead of "three").

### 1.2 Notes on the Problems
_Error detection_ for (1) seems trivial, often results in False Positives due to inexhaustive dictionaries, use of abbreviations, domain-specific terms and acronyms, etc. Moreover, a dictionary-based approach becomes more inefficient with more morphologically rich languages such as German. 

_Candiate generation_ can be approached naively through an approach similar to the one proposed by Peter Norvig (2007). This becomes infeasible for languages such as Mandarin, though, as it has 7,000 characters and operations like insertion and replacement would yield an exceedingly long candidate list. **(Note: check out how SymSpell manages candidate generation only with deletes)**

_Candidate selection_ is an almost equally difficult processes for both types of errors. Frequency lists and metrics such as the Levenshtein distance as well as some somewhat tacit rules (e.g. spelling corrections rarely seem to occur in the first letter of a word, therefore biasing candidate selection towards candidates that have the same starting letter). If we do, however, have the means to detect context-dependent errors like (2), we probably also have the means to use context as a factor in candidate selection.

### 1.3 Limiting the Scope of the Problem
While constraints of other languages are kept in mind during development of the individual approaches, the development and testing only occurrs on English data in order to limit the scope to a manageable level.


## 2. Dataset
As the dataset needs to be similar in format to the kind of data that spelling correction would be applied to in real-world scenarios, simple error dictionaries will not suffice. Instead, English data of whole sentences from a source without quality assurance will be used, as typos will likely occur in sources like that. Furthermore, the linguistic characteristics should be as general as possible, ruling out Twitter as a source due to the highly stylized format of Tweets. A publicly available dataset of 50,000 IMDB movie reviews from Kaggle (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/) was chosen.
This means, however, that we do not have labels on incorrect words, leaving us with no opportunity to calculate concrete metrics like accuracy on the individual approaches. Depending on the scope of this project down the line, manually labelled data could be genarated through Mechanical Turk or similar services.


## 3. Approach 1: Word Embeddings
In summary, this approach aims to generate a dictionary containing all typos of a given text (in our case, all typos in the dataset). The dictionary will contain the typo as a key and a list of potential corrections as its value. This approach is context-insensitive and requires a sufficiently large dataset to yield results. The proposed benefit is a significantly reduced number of False Positives compared to approaches that apply dictionary-based error detection, which makes this solution interesting for large datasets that are to be analyzed automatically for market research, where False Positives (and therefore risking altering the meaning of the data to be evaluated) are associated with much higher cost (i.e. are much less desireable) than False Negatives.

### 3.1 Error Detection
To avoid false positives, this approach will not rely solely on checking whether a word is in. Instead, we will generate word embeddings over the entire dataset, query for near neighbors of very frequent words (assuming that words that occur in high frequency are not misspelled) and check these candidates against a list of criteria to rule out that they are not proper words. A dict is generated, the incorrect word is added as a key and the word that was used to find the word is stored as a value.

### 3.2 Candidate Generation
The list of candidates consists of words that have a high similarity to the error in the word vector space as well as a limited Levenshtein distance.

### 3.3 Candidate Selection
**(Preliminary! Results are pretty awful. Perhaps order should be: close Levenshtein --> close vector)** The candidate selection occurs based on the first element in the list. This means that words that occur with higher frequency in the data set have a higher chance of being the first candidate, as they are queried for first.


## Sources
Norvig, Peter. (2007). _How to Write a Spelling Corrector._ Retrieved from https://norvig.com/spell-correct.html (30.07.2020)