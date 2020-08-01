import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from settings import FEAT_SIZE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('loading dfs')
embeds = pd.read_csv('data/embeddings.csv')
scores = pd.read_csv('data/adjusted_scores.csv')

print('merging...')
df = embeds.merge(scores, on='comment_id')

del embeds
del scores

print('Converting strings...')
df['embedding'] = df['embedding'].apply(lambda x: [float(s) for s in x.split()])


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

#df = df[df['score'] > 1]
#df['score'] = normalize(np.log(df['score']))


X = []
y = []
for x in df['embedding']:
    X.append(x)
for s in df['score']:
    y.append(s)

X = np.array(X)
y = np.array(y)

lmod = sm.OLS(y, sm.add_constant(X))
lmod_res = lmod.fit()
print(lmod_res.summary())

for i in range(len(X)):
    r = np.random.randint(len(X))
    X[i], X[r] = X[r], X[i]
    y[i], y[r] = y[r], y[i]

X_test, y_test = X[:len(X)//5], y[:len(y)//5]
X_train, y_train = X[len(X)//5:], y[len(y)//5:]

X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = np.array(X_train), np.array(y_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(FEAT_SIZE, kernel_initializer=tf.keras.initializers.HeNormal()),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  #tf.keras.layers.Dense(1024, activation='relu'),
  #tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])
es = tf.keras.callbacks.EarlyStopping(monitor='mse', mode='min', verbose=1, patience=5)
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=1, callbacks=[es])
mse_hist = history.history['mse']

sigmasqr = np.var(y_test)
sigmasqr_alt = np.mean((model.predict(X_test) - y_test) ** 2)
print(sigmasqr)
print(f'R^2 = {1 - sigmasqr_alt / sigmasqr}')

preds = np.array([v[0] for v in model.predict(X)])
df['score'] = normalize(preds - df['score'].values)

#df[['comment_id', 'score']].to_csv('data/adjusted_scores.csv', index=False)