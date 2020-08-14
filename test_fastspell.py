import pytest
import fastspell

"""
Test class for basic functionality of the FastSpell spelling corrector.
"""

def test_vocabulary():
    """
    Tests that frequency list contains exactly the same tokens as word 
    embeddings.
    """
    fs = fastspell.FastSpell("data/pnlp_data.csv")
    for word in fs.frequency_list.elements:
        assert word in fs.embeddings.wv.vocab
