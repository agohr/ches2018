import numpy as np


from matplotlib import pyplot as plt

def occlusion(traces, window, model, batch_size=100):
  t = np.copy(traces);
  Z0 = model.predict(t, batch_size=batch_size);
  t = t.transpose(); t[window] = 0; t = t.transpose();
  Z = model.predict(t,batch_size=batch_size);
  diff = Z - Z0;
  diff = diff*diff;
  d_average = np.mean(diff,axis=0);
  return(d_average);


