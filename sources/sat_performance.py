import numpy as np
import aes_ks_sat as aks

from math import floor, ceil
from time import time
from random import sample

def test_sat_performance(Z, keys, dropout=20):
  diff = np.abs(Z - keys);
  wrong = (diff > 1); w = np.sum(wrong,axis=1);
  counter = 0;
  n = len(Z);
  wt_good = 0.0; wt_bad = 0.0; arr_time = np.zeros(n); arr_sat = np.zeros(n,dtype=np.bool);
  for i in range(n):
    t0 = time();
    top2 = [(int(floor(x)), int(ceil(x))) for x in Z[i]];
    drop = sample([j for j in range(176) if not wrong[i][j]], dropout-w[i]);
    drop = np.array([(j in drop) for j in range(176)]);
    print(drop.shape, wrong.shape);
    drop = drop + wrong[i];
    sat, sol = aks.solve_trace(top2, dropout=drop);
    t1 = time(); dt = t1 - t0;
    print(i, sat, dt);
    if (sat): counter = counter + 1;
    if (sat): wt_good += dt; 
    else: wt_bad += dt;
    arr_time[i] = dt; arr_sat[i] = sat;
  t1 = time();
  return(counter, n, (arr_time, arr_sat));

