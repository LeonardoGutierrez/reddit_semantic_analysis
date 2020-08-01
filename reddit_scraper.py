import praw
import pandas as pd
from tqdm import tqdm
import numpy as np
import os 

from settings import NUM_SUBREDDITS, NUM_SUBMISSIONS, NUM_REPLACEMORE, TIME_FILTER, RAW_DIR, IGNORE_EDITED, MIN_SCORE

reddit = praw.Reddit(client_id="5_eiyQyg7Ne_xA",
                     client_secret="jR1blILFeBY9kALx0q4YHGMPdcg",
                     user_agent="Ny Scraper")


subreddit_dict = {'subreddit_id': [], 'subreddit_name': [], 'subreddit_subs': []}
submission_dict = {'submission_id': [], 'submission_score': []}
comment_dict = {'comment_id': [], 'parent_id': [], 'body': [], 'score': [], 'time_delta': []}

subreddit_submission_dict = {'subreddit_id': [], 'submission_id': []}
submission_topcomment_dict = {'submission_id': [], 'comment_id': []}

for isr, subreddit in enumerate(reddit.subreddits.popular(limit=NUM_SUBREDDITS+1)):
    if subreddit.display_name == 'Home':
        continue
    print(f"Working on subreddit ({isr}/{NUM_SUBREDDITS}): {subreddit.display_name}")
    subreddit_dict['subreddit_id'].append(subreddit.name)
    subreddit_dict['subreddit_name'].append(subreddit.display_name)
    subreddit_dict['subreddit_subs'].append(subreddit.subscribers)

    #Setting the time filter to the current day eliminates 'growth' confounding variables.
    for isb, submission in enumerate(subreddit.top(limit=NUM_SUBMISSIONS, time_filter=TIME_FILTER)):
        print(f"    Working on submission ({isb+1}/{NUM_SUBMISSIONS}): {submission.title[:40]}...")

        if submission.score < 2:
            continue

        submission_dict['submission_id'].append(submission.id)
        submission_dict['submission_score'].append(submission.score)

        subreddit_submission_dict['subreddit_id'].append(subreddit.name)
        subreddit_submission_dict['submission_id'].append(submission.id)

        submission.comments.replace_more(limit=NUM_REPLACEMORE)
        for topcom in submission.comments:
            if (topcom.edited and IGNORE_EDITED) or topcom.score < MIN_SCORE or topcom.body == '[removed]':
                continue
            comment_dict['comment_id'].append(topcom.id)
            comment_dict['body'].append(topcom.body)
            comment_dict['score'].append(topcom.score)
            comment_dict['time_delta'].append(topcom.created_utc - submission.created_utc)
            comment_dict['parent_id'].append(None)

            submission_topcomment_dict['submission_id'].append(submission.id)
            submission_topcomment_dict['comment_id'].append(topcom.id)

            for reply in topcom.replies:
                if (reply.edited and IGNORE_EDITED) or reply.score < MIN_SCORE or reply.body == '[removed]':
                    continue
                comment_dict['comment_id'].append(reply.id)
                comment_dict['body'].append(reply.body)
                comment_dict['score'].append(reply.score)
                comment_dict['parent_id'].append(topcom.id)
                comment_dict['time_delta'].append(reply.created_utc - topcom.created_utc)



subreddit_df = pd.DataFrame(subreddit_dict)
submission_df = pd.DataFrame(submission_dict)
comment_df = pd.DataFrame(comment_dict)
subreddit_submission_df = pd.DataFrame(subreddit_submission_dict)
submission_topcomment_df = pd.DataFrame(submission_topcomment_dict)
print(subreddit_df)
print(submission_df)
print(comment_df)
print(subreddit_submission_df)
print(submission_topcomment_df)

SUBDIR = 'associations/'

try:
    os.mkdir(RAW_DIR)
except:
    pass

try:
    os.mkdir(RAW_DIR + SUBDIR)
except:
    pass

subreddit_df.to_csv(RAW_DIR + 'subreddits.csv', index=False)
submission_df.to_csv(RAW_DIR + 'submissions.csv', index=False)
comment_df.to_csv(RAW_DIR + 'comments.csv', index=False)
subreddit_submission_df.to_csv(RAW_DIR + SUBDIR + 'subreddit_submission.csv', index=False)
submission_topcomment_df.to_csv(RAW_DIR + SUBDIR + 'submission_topcomment.csv', index=False)