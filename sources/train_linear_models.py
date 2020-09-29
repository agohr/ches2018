from train_nets import laden
from aes_ks_analysis import hw

from sklearn.linear_model import Ridge, RidgeCV

from pickle import dump

import numpy as np

lin1 = RidgeCV(alphas=[2**i for i in range(5,15)]);

X = np.load('train_traces_shuffled.npy');
k123 = np.load('train_keys_shuffled.npy');

lin1.fit(X,k123);

dump(lin1, open('linear_model_trained_on_3_samples.p','wb'),protocol=2);

