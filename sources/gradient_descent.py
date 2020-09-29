#given some inputs, a neural network and target outputs, find perturbed inputs that will yield the target outputs

import numpy as np 

from keras.models import Model
from keras.layers import Input, Subtract, Flatten, Multiply, GlobalAveragePooling1D, Dot
from keras import backend as K


#Implement projected gradient descent.
#The mask parameter can be used to freeze parts of the trace (in this case, make mask a binary-valued vector as long as the traces).
#Freezing parts of the trace is useful for experiments aimed at seeing if the neural network can be made to produce the target prediction
#by changing only values that one believes not to contain sensitive values relative to the prediction target.
#Inputs should be a vector of traces x0, neural network net, and target values y.
#Processing 100 traces (i.e. x0 contains 100 traces, y contains 100 sets of target values) in one batch should be possible on a wide range of systems.

def pgd(x0,net, y, max_iter=20, step_size = 0.01, radius=150.0, mask=1.0, decay=1.0):
  x = np.copy(x0); n = len(x0); xt = np.copy(x0);
  inp = Input(shape=y[0].shape);
  sub1 = Subtract()([net.output, inp]);
  out = Dot(1)([sub1, sub1]);
  model = Model(inputs=[net.input, inp],outputs=out);
  grads = K.gradients(model.output, net.input)[0];
  grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5);
  iterate = K.function([net.input, inp], [model.output, grads]);
  for i in range(max_iter):
    loss_value, grads_value = iterate([x, y]);
    xt -= step_size * mask * grads_value;
    dx = xt - x0;
    s = np.sqrt(np.sum(dx * dx, axis=1));
    for j in range(n):
      if (s[j] > radius):
        xt[j] = x0[j] + dx[j] * (radius / s[j]);
    x = np.copy(xt);
    print(i, np.mean(loss_value/176));
    step_size = step_size * decay;
  return(x, loss_value);

def calc_gradient(net, x0, mask):
  inp = Input(shape=(176,));
  dotlayer = Dot(1)([inp, net.output]);
  model = Model(inputs=[inp, net.input], outputs=dotlayer);
  grads = K.gradients(model.output, net.input)[0];
  f = K.function([inp, model.input], [grads]);
  dZ = f([mask, x0])[0];
  return(dZ);


