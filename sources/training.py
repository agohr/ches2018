from pickle import load
from aes_ks_analysis import hw
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau#
from utils import shuffle_together, hw
from keras import backend as K
import numpy as np


import net_ches_2018 as nc

def train_net(depth=1, name_prefix='net_depth', working_dir = './nets/', num_epochs=5000):
  name = working_dir + name_prefix + str(depth) + '.h5';
  check = ModelCheckpoint(name,monitor='val_loss', save_best_only=True, save_weights_only=True);
  reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=50, factor=0.5, min_lr=0.00001);

  model = nc.make_net(depth=depth,reg_term=10**-5);
  model.compile(optimizer='adam',loss='mse',metrics=['acc']);
  
  #load random-key training traces after reduction to 65000 points and shuffling
  #also load S6 challenge traces for testing purposes
  traces = np.load('train_traces_shuffled.npy');
  keys = np.load('train_keys_shuffled.npy');
  X_test = np.load('s6_traces.npy');
  Y_test = np.load('s6_key.npy');
  Y_test = hw(Y_test);
  kh = hw(keys);
  h = model.fit(traces, kh, epochs=num_epochs, batch_size=100,validation_split=0.1,callbacks=[check,reduce_lr], verbose=2);
  model.load_weights(name);
  Z = model.predict(X_test);
  diff = np.abs(Z - Y_test);
  v = np.sum(diff > 1, axis=1);
  median, mean = np.median(v), np.mean(v);
  loss = np.min(h.history['val_loss']);
  return(median, mean, loss);

l = [1,2,5,10,15,19];

for i in l:
  median, mean, loss = train_net(depth=i, num_epochs=1000);
  print("Depth:", i, "Median errors: ", median, "Mean errors: ", mean, "Validation loss: ", loss)
  
  
