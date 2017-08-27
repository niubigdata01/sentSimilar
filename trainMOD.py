import logging
from gensim.models.word2vec import LineSentence, Word2Vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences= LineSentence("trainOneLine_Search.txt")

"""如果要再訓練更多句子   失敗
model = Word2Vec.load('./Mod/w2vtestSeverateLinesIt1.mod')
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
"""

model = Word2Vec(sentences, min_count=1, iter=50000)
model.save("w2vFinishIt50000.mod")
