from gensim.models import Word2Vec
import pandas as pd
from settings import SENT_SEP, DATA_DIR, FEAT_SIZE
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

NUM_EPOCH = 50
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} {self.epoch / NUM_EPOCH * 100}%")
        self.epoch += 1
epoch_logger = EpochLogger()

df = pd.read_csv(DATA_DIR + 'clean_comments.csv')
sents = []

print('reading in comments')
for comment in tqdm(df['body']):
    comsents = comment.split(SENT_SEP)
    comsents = [com.split() for com in comsents]
    sents.extend(comsents)

print('creating model')
model = Word2Vec(sents, size=FEAT_SIZE, window=10, min_count=3, iter=NUM_EPOCH, workers=16, callbacks=[epoch_logger])

print('saving model...')
model.callbacks = ()
model.save(DATA_DIR + "word2vec.model")