import numpy as np
import h5py
import aes_ks

def hw(x):
  res = np.zeros(x.shape, dtype=np.uint8);
  for i in range(8):
    res = res + ((x >> i) & 1);
  return(res);

def shuffle_together(l):
  state = np.random.get_state();
  for x in l:
    np.random.set_state(state);
    np.random.shuffle(x);

def low_pass(data, cutoff):
  tmp = np.fft.fft(data);
  n = data.shape[::-1][0];
  tmp[:,cutoff:n-cutoff+1] = 0;
  res = np.fft.ifft(tmp);
  return(res);

def high_pass(data, cutoff):
  tmp = np.fft.fft(data);
  n = data.shape[::-1][0];
  tmp[:,1:cutoff] = 0; tmp[:,n-cutoff+1:n] = 0;
  res = np.fft.ifft(tmp);
  return(res);
  
def test_performance(data, keys, model):
  Z = model.predict(data);
  Z = Z[:,0:176];
  diff = np.abs(Z - keys);
  v = np.sum(diff > 1, axis=1);
  mse = np.mean(diff*diff);
  return(v, mse);

#shift trace by k steps to the right 
#fill the first few entries by leaving them as is
def shift_trace(s, k):
  n = s.shape[1];
  res = np.copy(s);
  res[:,k:] = s[:,0:n-k];
  return(res);

def decimate_trace(s, k):
  n = s.shape[1];
  res = s[:,0:n:k];
  return(res);

def expand_all_keys(keys):
  kres = np.zeros((len(keys),176),dtype=np.uint8);
  for i in range(len(keys)):
    kres[i] = aes_ks.ks_expand(keys[i]);
  return(kres);

def process_trace_files(l, tracefile='train_traces_shuffled.npy', keyfile = 'train_keys_shuffled.npy'):
  X = []; Y = [];
  for f in l:
    tmp = h5py.File(f);
    Xtmp = np.array(tmp['0']['samples']['value']);
    Xtmp = decimate_trace(Xtmp, 10);
    ktmp = np.array(tmp['0']['key']['value']).astype(np.uint8);
    k = expand_all_keys(ktmp);
    X = X + [Xtmp];
    Y = Y + [k];
  X = np.concatenate(X); Y = np.concatenate(Y);
  shuffle_together([X,Y]);
  np.save(tracefile, X); np.save(keyfile,Y);
    
        
  
