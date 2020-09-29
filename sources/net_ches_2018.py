from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Add, Conv1D, Reshape, AveragePooling1D, Flatten, Dropout
from keras.regularizers import l2

def make_net(q1=100,q2=650, depth=19,reg_term=10**-5):
    p = q1 * q2;
    inp = Input(shape=(p,));
    res1 = Reshape((q1,-1))(inp);
    bn = BatchNormalization()(res1);
    conv = Conv1D(176,1,activation='relu',padding='same',kernel_regularizer=l2(reg_term))(bn);
    shortcut = conv;
    for i in range(depth):
        bn = BatchNormalization()(conv);
        conv = Conv1D(176,3,activation='relu',padding='same',kernel_regularizer=l2(reg_term))(bn);
        conv = Add()([conv, shortcut]);
        shortcut = conv;
    out = AveragePooling1D(pool_size=q1)(conv);
    out = Flatten()(out);
    model = Model(inputs=inp, outputs=out);
    return(model);


