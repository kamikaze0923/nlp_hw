import numpy as np
import itertools

def s(x):
  return 1/(1 + np.exp(-x))

################################
# Task 1.2
################################

# i gate
w_ix = 0
w_ih = -100
b_i = 38

# f gate
w_fx = -76
w_fh = 100
b_f = -38

# o gate
w_ox = 0
w_oh = 0
b_o = 38

# g
w_gx = 38
w_gh = 0
b_g = 0


################################

# The below code runs through all length 14 binary strings and throws an error 
# if the LSTM fails to predict the correct parity

cnt = 0
for X in itertools.product([0,1], repeat=14):
  c=0 # two states(even, odd) for c. 0 and 1
  h=0 # two states(even, odd) for h. 0 and 0.76
  cnt += 1
  for x in X:
    i = s(w_ih*h + w_ix*x + b_i) # always accept input when h=0. always reject input when h=0.76
    f = s(w_fh*h + w_fx*x + b_f) # always forget when h=0. always remember input when h=0.76 and x=0. forget when h=0.76 and x=1
    g = np.tanh(w_gh*h + w_gx*x + b_g) # only used when c=0, set this to 1 to update c
    o = s(w_oh*h + w_ox*x + b_o) # always output
    c = f*c + i*g # when c=0, always forget current c and update from input, when c=1 update by forget itself(controlled by x and h both)
    h = o*np.tanh(c)# tanh(1) = 0.76 will trigger, so try to keep c=0 or 1
  if np.sum(X)%2 != int(h>0.5):
    print("Failure",cnt, X, int(h>0.5), h, np.sum(X)%2 == int(h>0.5))
    break
  if cnt % 1000 == 0:
    print(cnt)
