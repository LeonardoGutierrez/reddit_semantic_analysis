import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize.casual import TweetTokenizer
from collections import defaultdict, Counter
import functools
import operator
import re
from tqdm import tqdm
from settings import RAW_DIR, DATA_DIR, SENT_SEP


tweet_tokenizer = TweetTokenizer()
def clean_text(text):
    out = []
    for sent in sent_tokenize(text.lower()):
        words = tweet_tokenizer.tokenize(sent)
        out.extend(words)
        out.append(SENT_SEP)
    del out[-1]
    return ' '.join(out)

def tfidf(text, vocab_idf):
    flatten = [word for word in text.replace(SENT_SEP, ' ').split()]
    counter = Counter(flatten)
    return ' '.join([str(counter[word] * vocab_idf[word] / len(flatten)) for word in flatten])


print("loading raw comments")
df = pd.read_csv(RAW_DIR + 'comments.csv')
df = df[df['body'].notna()]


tqdm.pandas()
print('cleaning text')
df['body'] = df['body'].progress_apply(lambda x: clean_text(x))


print('calculating idf')
vocab_docfreq = defaultdict(int)
num_docs = 0
for _, text in tqdm(df['body'].iteritems()):
    num_docs += 1
    for w in set([word for word in text.replace(SENT_SEP, ' ').split()]):
        vocab_docfreq[w] += 1

vocab_idf = dict()
for k in vocab_docfreq.keys():
    vocab_idf[k] = np.log(num_docs / vocab_docfreq[k])

idf_df = pd.DataFrame.from_dict(vocab_idf, orient='index', columns=['idf'])
idf_df.reset_index(level=0, inplace=True)
idf_df.rename(columns={'index': 'word'}, inplace=True)

print('saving idf.csv')
idf_df.to_csv(DATA_DIR + 'idf.csv', index=False)

print('calculating tfidf')
df['tfidf'] = df['body'].progress_apply(lambda x: tfidf(x, vocab_idf))

df = df[df['body'].notna()]
df = df[df['tfidf'].notna()]
df = df[df['body'] != '']
df = df[df['tfidf'] != '']

print('saving clean_comments.csv')
df[['comment_id', 'body', 'tfidf']].to_csv(DATA_DIR + 'clean_comments.csv', index=False)

