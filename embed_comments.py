from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from settings import DATA_DIR, SENT_SEP, FEAT_SIZE
from tqdm import tqdm


clean_df = pd.read_csv(DATA_DIR + 'clean_comments.csv')
model = Word2Vec.load(DATA_DIR + 'word2vec.model')


tqdm.pandas()
print('reading in tfidfs')
clean_df['tfidf'] = clean_df['tfidf'].progress_apply(lambda x: [float(s) for s in x.split()] if isinstance(x, str) else [float(x)])

def embed(comment, tfidfs):
    global model
    sents = comment.split(SENT_SEP)

    tf_ind = 0
    post_vec = np.zeros(FEAT_SIZE)
    for sent in sents:
        if len(sent) == 0:
            continue
        vec = np.zeros(FEAT_SIZE)
        for w in sent.split():
            try:
                vec += model.wv[w] * tfidfs[tf_ind]
            except:
                pass
            tf_ind += 1
        #vec /= len(sent)
        post_vec += vec
    #post_vec /= len(sents)

    return ' '.join([str(f) for f in post_vec])

print('creating embeddings...')
clean_df['embedding'] = clean_df.progress_apply(lambda x: embed(x['body'], x['tfidf']), result_type='reduce', axis=1)

print('saving embeddings.csv')
clean_df[['comment_id', 'embedding']].to_csv(DATA_DIR + 'embeddings.csv', index=False)