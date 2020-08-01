from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

model = Word2Vec.load('data/word2vec.model')




pprint(model.wv.most_similar('reddit'))
pprint(model.wv.most_similar('trump'))
pprint(model.wv.most_similar('politics'))
pprint(model.wv.most_similar('bezos'))
pprint(model.wv.most_similar('homeless'))