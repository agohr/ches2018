import aes_ks
from pycryptosat import Solver
import numpy as np
from random import random, randint

from pickle import load

from time import time

from random import sample

from multiprocessing import Pool


#take a boolean function and a list of input variables and output the cnf of the function
#last variable is the output
def sign(j,n):
  return((-1) ** ((n >> j) & 1));

def create_cnf(f, varlist):
  n = 2 ** (len(varlist)-1);
  clauses = [];
  for i in range(n):
    out = f[i];
    out = not out;
    cl = [sign(j,i) * varlist[j] for j in range(len(varlist)-1)];
    cl = cl + [((-1) ** out) * varlist[len(varlist)-1]];
    clauses = clauses + [cl];
  return(clauses);

#same as before, but no explicit output: the cnf will just describe the set of inputs evaluating to true
def create_cnf_true(f, varlist):
  n = 2 ** (len(varlist));
  clauses = [];
  bad = [i for i in range(n) if not f[i]];
  for i in bad:
    cl = [sign(j,i) * varlist[j] for j in range(len(varlist))];
    clauses = clauses + [cl];
  return(clauses);

def hw(x, k):
  s = np.zeros(x.shape, dtype=np.uint8);
  for i in range(k):
    s = ((x >> i) & 1) + s;
  return(s);

def restrict_hw(varlist, f):
  x = np.array(range(2**len(varlist)),dtype=np.uint8);
  #print(x);
  hx = hw(x, len(varlist));
  #print(hx)
  b = np.vectorize(f)(hx);
  #print(b);
  clauses = create_cnf_true(b, varlist);
  return(clauses);

def restrict_hw_soft(varlist, f):
  x = np.array(range(2**(len(varlist)-1)), dtype=np.uint8);
  hx = hw(x, len(varlist));
  b = np.vectorize(f)(hx);
  b2 = np.ones(2 ** (len(varlist)-1));
  b = np.concatenate([b,b2]);
  clauses = create_cnf_true(b, varlist);
  return(clauses);

def capped_sum(varlist1, varlist2, varlist_out):
  mx = 2 ** len(varlist_out) - 1;
  a = 2 ** len(varlist1); b = 2 ** len(varlist2);
  clauses = [];
  for i in range(len(varlist_out)):
    out = varlist_out[i];
    l = varlist1 + varlist2 + [out];
    n = 2 ** (len(varlist1) + len(varlist2));
    x = np.array(range(n));
    for j in range(n):
      z = (j & (a-1)) + (j >> len(varlist1));
      if (z > mx): z = mx;
      x[j] = (z >> i) & 1;
    clauses = clauses + create_cnf(x, l);
  return(clauses); 

def leq(varlist, limit):
  f = [(x <= limit) for x in range(2 ** len(varlist))];
  clauses = create_cnf_true(f, varlist);
  return(clauses);    

#restrict the variables in varlist_out to aes sbox outputs for inputs in varlist_in 
def sbox(varlist_in, varlist_out):
  clauses = [];
  for i in range(len(varlist_out)):
    sb = (aes_ks.sbox >> i) & 1;
    v = list(varlist_in) + [varlist_out[i]];
    clauses = clauses + create_cnf(sb,v);
  return(clauses);

#restrict the variables in varlist to having bitwise sum zero (or constant if constant is supplied)
def zero(varlist, c=0):
  return(restrict_hw(varlist, lambda x: (x & 1) == c));

#construct aes key schedule with k rounds
#we will need (k+1) * 16 * 8 variables for the regular key schedule vars
#plus 32 variables per round (32(k+1)) helper variables for the s-box outputs
def rotate(j):
  tmp = j + 1;
  if (tmp > 3): tmp = 0;
  return(tmp);

def aes_key_schedule(k, with_sbox = True, sb_drop_rate = 0.0):
  clauses = [];
  num_vars = (k+1) * 16 * 8 + 32 * (k+1);
  helper_offset = ((k+1) * 16 * 8);
  #set the helper variables to be the s-box outputs of the corresponding regular variables
  for i in range(k+1):
    for j in range(4):
      v_in = range(16 * 8 * i + 12 * 8 + 8 * rotate(j) + 1, 16 * 8 * i + 12 * 8 + 8 * rotate(j) + 8 + 1);
      v_out = range(helper_offset + 32 * i + 8 * j + 1, helper_offset + 32 * i + 8 * j + 8 + 1);
      if with_sbox: clauses = clauses + sbox(v_in, v_out);
      else:
        if random() > sb_drop_rate: clauses = clauses + sbox(v_in, v_out); 
  #set the first four bytes of subkeys 1..k to be the bitwise sum of subkeys 0..k-1 and the helpers
  for i in range(0,k):
    v_in1 = range(16 * 8 * i + 1, 16 * 8 * i + 32 + 1);
    v_in2 = range(helper_offset + 32 * i + 1, helper_offset + 32 * i + 32 + 1);
    v_out = range(16 * 8 * (i+1) + 1, 16 * 8 * (i+1) + 32 + 1);
    rcon = aes_ks.rcon[i+1]; rc = [(rcon >> j) & 1 for j in range(32)];
    for a,b,c, r in zip(v_in1, v_in2, v_out, rc):
      clauses = clauses + zero([a,b,c],r);
  #set the remaining 12 bytes of subkeys 1..k to be the bitwise sum of subkeys 0..k-1 and the bytes of subkey 1..k immediately preceding
  for i in range(0,k):
    v_in1 = range(16 * 8 * i + 32 + 1, 16 * 8 * i + 128 + 1);
    v_out = range(16 * 8 * (i+1) + 32 + 1, 16 * 8 * (i+1) + 128 + 1);
    v_in2 = range(16 * 8 * (i+1) + 1, 16 * 8 * (i+1) + 96 + 1);
    for a,b,c in zip(v_in1, v_in2, v_out):
      clauses = clauses + zero([a,b,c]);
  return(clauses);

def restrict_hw_in_aes_ks(weights, dropout=0.0):
  clauses = [];
  for i in range(len(weights)):
    b = range(8 * i + 1, 8 * i + 9);
    r = random();
    if (r > dropout): clauses = clauses + restrict_hw(b, lambda x: x == weights[i]);
  return(clauses);

def restrict_hw_in_aes_ks_fuzzy(weights, dropout=[False for i in range(176)], wr_dropout=1.0):
  clauses = [];
  for i in range(len(weights)):
    b = range(8 * i + 1, 8 * i + 9);
    r = random();
    if (not dropout[i]): clauses = clauses + restrict_hw(b, lambda x: x in weights[i]);
    else:
      clauses_new = restrict_hw(b, lambda x: x in weights[i]);
      rand_arr = np.random.rand(len(clauses_new));
      clauses_new = [clauses_new[i] for i in range(len(clauses_new)) if rand_arr[i] > wr_dropout];
      clauses = clauses + clauses_new;
  return(clauses);

def convert_to_byte(l):
  return(sum([2**i for i in range(len(l)) if l[i]]));

#Game plan: load guessed hamming weights for a trace. Pick bytes to drop. Generate CNF for the key schedule using aes_key_schedule. Generate CNF for hamming weight restrictions using restrict_hw_in_aes_ks. Get an instance of the pycryptosat solver. Feed it both CNFs. Solve. Convert back to byte array.

#solve trace with given dropout
#dropout means that the given part of the supplied hamming weights will not be used
def solve_trace(data, dropout=[False for i in range(176)], cfl = 300000, wr_dropout=1.0):
  keyschedule = aes_key_schedule(10);
  weight_restrictions = restrict_hw_in_aes_ks_fuzzy(data, dropout=dropout, wr_dropout=wr_dropout);
  solver = Solver(confl_limit=cfl);
  for clause in keyschedule: solver.add_clause(clause);
  for clause in weight_restrictions: solver.add_clause(clause);
  print("Number of clauses: ", len(keyschedule)+len(weight_restrictions));
  sat,sol = 0,1;
  try:
    sat, sol = solver.solve();
  except:
    sat = None   
  if (not sat):
    print("No success satisfying system.");
    return(False, None);
  #convert solution to keyschedule byte values
  byte_arr = np.array([convert_to_byte(sol[i:i+8]) for i in range(1,176*8,8)],dtype=np.uint8);
  return(True, byte_arr);

def correct_guesses(guesses, dropout=20, cfl=300000, drop_level = 2.01):
  drop = sample(range(176),dropout);
  d = np.full(176,1.01); d[drop] = drop_level;
  top = [[i for i in range(0,9) if abs(guesses[j] - i) < d[j]] for j in range(176)];
  t, s = solve_trace(top, cfl=cfl, dropout=[(i in drop) for i in range(176)]);
  return(t,s);

def correct_guesses_full(guesses, dropout=20, cfl=300000, drop_level=2.0, max_tries=100):
  for i in range(max_tries):
    t, s = correct_guesses(guesses, dropout=dropout, cfl=cfl, drop_level=drop_level);
    print(i, t, s);
    if t: return(t,s);
  return(False, None);

def f(x):
  guess = x[0]; drop = x[1]; cfl = x[2];
  return(solve_trace(guess, dropout=drop, cfl=cfl));
  
def test_cleanup_guesses(guesses, key, num_threads=12):
  sat = False;
  wrong = [i for i in range(176) if not (key[i] in guesses[i])];
  pool = Pool(num_threads);
  i = 0; cfl = 300000;
  while (not sat):
    i = i + 1;
    s = [sample(range(176),35) for j in range(num_threads)];
    print(s);
    print(wrong);
    drop = [[(j in s[j2]) for j in range(176)] for j2 in range(num_threads)];
    drop2 = [[j for j in range(176) if drop[j2][j]] for j2 in range(num_threads)];
    sat_in_principle = [(set(wrong) < set(x)) for x in s];
    sat_in_principle2 = [(set(wrong) < set(x)) for x in drop2];
    print([sum(x[0:100]) for x in drop], cfl);
    t0 = time();
    sols = pool.map(f, [(guesses,x, cfl) for x in drop]);
    sats = [x[0] for x in sols]; sols = [x[1] for x in sols];
    sat = sum(sats);
    t1 = time();
    print(i, t1-t0, sat, sats, sat_in_principle, sat_in_principle2);
    if (sat):
      ind = np.argmax(sats); sol = sols[ind]; 
      print(sol);
      return(sol);

def cleanup_guesses(guesses, num_threads=12,dropout=20, cfl = 300000):
  sat = False;
  pool = Pool(num_threads);
  i = 0;
  while (not sat):
    i = i + 1;
    s = [sample(range(176),dropout) for j in range(num_threads)];
    drop = [[(j in s[j2]) for j in range(176)] for j2 in range(num_threads)];
    t0 = time();
    sols = pool.map(f, [(guesses,x, cfl) for x in drop]);
    sats = [x[0] for x in sols]; sols = [x[1] for x in sols];
    sat = sum(sats);
    t1 = time();
    print(i, t1-t0, sat, sats);
    if (sat):
      ind = np.argmax(sats); sol = sols[ind]; 
      print(sol);
      return(sol);
    
#given a key, create a fake top2-guess for testing purposes
def fuzzyfy_solution(key):
  res = [(x, x + (2 * randint(0,2) - 1)) for x in key];
  return(res);

  
    
