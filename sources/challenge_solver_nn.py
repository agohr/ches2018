import sys

import h5py

import aes_ks_sat as aks
from keras.models import load_model 
from pickle import load
from math import floor, ceil
from random import random
from net_ches_2018 import make_net

import numpy as np

def stats(Z, key_hw, j,delta=1):
  n = len(Z);
  m = (n // j) * j;
  Z2 = np.copy(Z[0:m]);
  Z2 = Z2.reshape(m//j,j,-1);
  Z_av = np.mean(Z2, axis=1);
  #print(Z_av.shape); 
  diff = Z_av - key_hw;
  diff = np.abs(diff);
  print("Using ", j, " traces: ");
  v = np.sum(diff >= delta, axis=1); 
  #print(v); 
  res = [np.sum(v <= i).astype(np.float32) / (m / j) for i in range(6)]
  print(res);

def extract(X,net,n=10):
  k = len(X); res = np.zeros((n,k,176))
  for i in range(n):
    T = X.transpose()[i:650000:n].transpose();
    res[i] = net.predict(T,batch_size=500);
  Z = np.mean(res, axis=0);
  return(Z);

if (len(sys.argv) < 3):
  print('Script needs as arguments a traces file and a model');
  print('Mandatory arguments not supplied');
  sys.exit();

f = h5py.File(sys.argv[1]);

print("Reading traces...");

X = np.array(f['0']['samples']['value']);

X = X.transpose();
X = X[0:len(X):10].transpose();

model = load_model(sys.argv[2]);
print('Predicting hamming weights...');

Z = model.predict(X);
Z[Z < 0] = 0.01; Z[Z > 8] = 7.99;

num_traces_to_use = 0;
if (len(sys.argv) > 3):
  num_traces_to_use = int(sys.argv[3]);
else:
  num_traces_to_use = len(Z);

print("Using " + str(num_traces_to_use) + " traces.");

Z_av = np.median(Z[0:num_traces_to_use],axis=0);
Z_av2 = np.mean(Z[0:num_traces_to_use],axis=0);

top2 = [(int(floor(x)), int(ceil(x))) for x in Z_av2];

print('Solving system');

sol = aks.cleanup_guesses(top2[0:176],num_threads=1);

key = sol[0:16];

key_hw = aks.hw(sol[0:176],8);

diffs = Z - key_hw;
diffs = np.abs(diffs);
diffs = np.sum(diffs >= 1, axis=1);
print("Mean, median number of bad top-2 guesses for the 176 extended key bytes (single trace):");
print(np.mean(diffs),np.median(diffs));
print("Number of traces in the challenge set with less than 0,1,2,... bad top-2 guesses:");
q = [np.sum(diffs <= i) for i in range(6)];
print(q);

keyhex = [hex(x) for x in key];
print("Solution:");
print(keyhex);
s = '';
for x in key:
  s = s + "{0:0{1}x}".format(x,2);
print("As string: ", s);

print("Success rate statistics for this challenge, as a function of the number of bad guesses tolerated (each bad guess increases expected computation to solution by a factor of roughly nine):");
for i in range(1,6):
  stats(Z, key_hw, i);

