from gensim.models.fasttext import FastText
import fastspell

sc = fastspell.FastSpell("data/corpus.txt")
model = sc.embeddings

print(model.wv.most_similar("customer", topn=10))