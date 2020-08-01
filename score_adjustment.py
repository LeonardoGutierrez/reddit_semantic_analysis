import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA

from settings import RAW_DIR, DATA_DIR

subreddit_df = pd.read_csv(RAW_DIR + 'subreddits.csv')
submission_df = pd.read_csv(RAW_DIR + 'submissions.csv')
submission_topcomment_df = pd.read_csv(RAW_DIR + 'associations/submission_topcomment.csv')
subreddit_submission_df = pd.read_csv(RAW_DIR + 'associations/subreddit_submission.csv')

comment_df = pd.read_csv(RAW_DIR + 'comments.csv', usecols=['comment_id', 'score', 'parent_id', 'time_delta'])

df = subreddit_df.merge(subreddit_submission_df, on='subreddit_id')
df = df.merge(submission_df, on='submission_id')
df = df.merge(submission_topcomment_df, on='submission_id')
df = df.rename(columns={'comment_id': 'top_id'})


df = df.merge(comment_df, left_on='top_id', right_on='comment_id').drop(columns=['comment_id', 'parent_id'])
df = df.rename(columns={'time_delta': 'top_time_delta', 'score':'top_score'})

df = df.merge(comment_df, left_on='top_id', right_on='parent_id')
df = df.rename(columns={'time_delta': 'reply_time_delta', 'score':'reply_score'})
df = df.drop(columns=['parent_id']).rename(columns={'comment_id': 'reply_id'})

df = df[df['top_score'] > 2]
df = df[df['reply_score'] > 2]

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

data_df = pd.DataFrame()
data_df['ttd'] = df['top_time_delta'] ** 0.5  / 60
data_df['rtd'] = df['reply_time_delta'] ** 0.5 / 60
data_df['log_subreddit_subs'] = np.log(df['subreddit_subs'])
data_df['log_submission_score'] = np.log(df['submission_score'])
data_df['log_top_score'] = np.log(df['top_score'])

data_df.dropna(inplace=True)
print(data_df['log_top_score'])
subred_cats = sm.categorical(df['subreddit_name'], drop=True)



draw_df = data_df.copy()#pd.DataFrame(data = pca_comp)
draw_df['log_reply_score'] = np.log(df['reply_score'])
draw_df['subreddit_name'] = df['subreddit_name']
sns.pairplot(draw_df, markers='+', hue='subreddit_name', 
                      y_vars=['log_reply_score'],
                      x_vars=['ttd', 'rtd','log_subreddit_subs', 'log_submission_score', 'log_top_score'])
plt.show()

X = np.column_stack([
    np.ones(len(df)),
    data_df.values,
    subred_cats
])

w =normalize(np.log(df['top_score'])) + 0.2
lmod = sm.WLS(np.log(df['reply_score']), X, weights=1/(w**2))
lmod_res = lmod.fit()
print(lmod_res.summary())

out = pd.DataFrame()
out['comment_id'] = df['reply_id']
out['score'] = normalize(np.log(df['reply_score']) - lmod_res.predict())


sns.distplot(out['score'], bins=50)
plt.show()

out.to_csv(DATA_DIR + 'adjusted_scores.csv', index=False)