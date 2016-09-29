
# coding: utf-8

# In[4]:

from keras.layers.recurrent import Recurrent
from keras import backend as K
from keras import activations, initializations

if K._BACKEND == 'tensorflow':
    import tensorflow as tf


"""class BiLSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',bi=True, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.bi = bi
        super(BiLSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]
            
    def get_output(self,train=False):
        
        self.go_backwards = False
        R1 = Recurrent.get_output(self,train)
        if not self.bi:
            return R1
        self.go_backwards = True
        R2 = Recurrent.get_output(self,train)

        if self.return_sequences:
            R2 = R2[::,::-1,::]
        return R1/2 + R2 /2
        


    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))"""


# In[8]:

class BiLSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',close=False, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.close=close
        self.inner_activation = activations.get(inner_activation)
        super(BiLSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim,))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))
        
        if self.close:
            self.W_h =  self.init((self.output_dim, self.output_dim))
            self.b_h = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]
        if self.close:
            self.trainable_weights += [self.W_h,self.b_h]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
    def get_output(self,train=False):
        
        self.go_backwards = False
        R1 = Recurrent.get_output(self,train)

        self.go_backwards = True
        R2 = Recurrent.get_output(self,train)

        if self.return_sequences:
            if K._BACKEND == 'tensorflow':
                R2 = tf.reverse(R2,[False,True,False])
            else:
                R2 = R2[::,::-1,::]
        if self.close:
            return  K.dot(R1 + R2 ,self.W_h) + self.b_h
        else:
            return  R1 / 2 + R2 / 2

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[13]:

import keras
if  int(keras.__version__.split(".")[0]) >= 1.0 :
    print "v1"
    from keras import activations, initializations, regularizers
    from keras.engine import Layer, InputSpec
    from keras.layers.recurrent import time_distributed_dense


    class BiLSTMv1(Recurrent):
        '''Long-Short Term Memory unit - Hochreiter 1997.

        For a step-by-step description of the algorithm, see
        [this tutorial](http://deeplearning.net/tutorial/lstm.html).

        # Arguments
            output_dim: dimension of the internal projections and the final output.
            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            forget_bias_init: initialization function for the bias of the forget gate.
                [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
                recommend initializing with ones.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            inner_activation: activation function for the inner cells.
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the input weights matrices.
            U_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
            dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

        # References
            - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''
        def __init__(self, output_dim,
                     init='glorot_uniform', inner_init='orthogonal',
                     forget_bias_init='one', activation='tanh',
                     inner_activation='hard_sigmoid',
                     W_regularizer=None, U_regularizer=None, b_regularizer=None,
                     dropout_W=0., dropout_U=0.,close=False, **kwargs):
            self.output_dim = output_dim
            self.init = initializations.get(init)
            self.inner_init = initializations.get(inner_init)
            self.forget_bias_init = initializations.get(forget_bias_init)
            self.activation = activations.get(activation)
            self.inner_activation = activations.get(inner_activation)
            self.W_regularizer = regularizers.get(W_regularizer)
            self.U_regularizer = regularizers.get(U_regularizer)
            self.b_regularizer = regularizers.get(b_regularizer)
            self.dropout_W, self.dropout_U = dropout_W, dropout_U
            self.close=close

            if self.dropout_W or self.dropout_U:
                self.uses_learning_phase = True
            super(BiLSTMv1, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            input_dim = input_shape[2]
            self.input_dim = input_dim

            if self.stateful:
                self.reset_states()
            else:
                # initial states: 2 all-zero tensors of shape (output_dim)
                self.states = [None, None]

            self.W_i = self.init((input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            if self.close:
                self.W_h =  self.init((self.output_dim, self.output_dim),
                                      name='{}_W_h'.format(self.name))
                self.b_h = K.zeros((self.output_dim,),
                                   name='{}_b_h'.format(self.name))

            self.regularizers = []
            if self.W_regularizer:
                if not self.close:
                    self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                                self.W_f,
                                                            self.W_c,
                                                            self.W_o]))
                if self.close:
                     self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                            self.W_f,
                                                            self.W_c,
                                                            self.W_o,
                                                            self.W_h]))

                self.regularizers.append(self.W_regularizer)
            if self.U_regularizer:
                self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                            self.U_f,
                                                            self.U_c,
                                                            self.U_o]))
                self.regularizers.append(self.U_regularizer)
            if self.b_regularizer:
                if not self.close:
                    self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                                self.b_f,
                                                                self.b_c,
                                                                self.b_o]))
                else:
                    self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                                self.b_f,
                                                                self.b_c,
                                                                self.b_o,
                                                               self.b_h]))
                self.regularizers.append(self.b_regularizer)

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]
            if self.close:
                self.trainable_weights += [self.W_h,self.b_h]


            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights

        def reset_states(self):
            assert self.stateful, 'Layer must be stateful.'
            input_shape = self.input_spec[0].shape
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided (including batch size).')
            if hasattr(self, 'states'):
                K.set_value(self.states[0],
                            np.zeros((input_shape[0], self.output_dim)))
                K.set_value(self.states[1],
                            np.zeros((input_shape[0], self.output_dim)))
            else:
                self.states = [K.zeros((input_shape[0], self.output_dim)),
                               K.zeros((input_shape[0], self.output_dim))]

        def preprocess_input(self, x, train=False):
            if self.consume_less == 'cpu':
                if train and (0 < self.dropout_W < 1):
                    dropout = self.dropout_W
                else:
                    dropout = 0
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[2]
                timesteps = input_shape[1]

                x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                             input_dim, self.output_dim, timesteps)
                return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
            else:
                return x


        def call(self,x,mask=None):

            self.go_backwards = False
            R1 = Recurrent.call(self,x,mask=mask)

            self.go_backwards = True
            R2 = Recurrent.call(self,x,mask=mask)

            if self.return_sequences:
                if K._BACKEND == 'tensorflow':
                    R2 = tf.reverse(R2,[False,True,False])
                else:
                    R2 = R2[::,::-1,::]
            if self.close:
                return  K.dot(R1 + R2 ,self.W_h) + self.b_h
            else:
                return  R1 / 2 + R2 / 2
            
        def cell(self,x):
            pass

        def step(self, x, states):
            h_tm1 = states[0]
            c_tm1 = states[1]
            B_U = states[2]
            B_W = states[3]

            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            else:
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

            h = o * self.activation(c)
            return h, [h, c]

        def get_constants(self, x):
            constants = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * self.output_dim, 1)
                B_U = [K.dropout(ones, self.dropout_U) for _ in range(4)]
                constants.append(B_U)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])

            if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[-1]
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)
                B_W = [K.dropout(ones, self.dropout_W) for _ in range(4)]
                constants.append(B_W)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])
            return constants

        def get_config(self):
            config = {"output_dim": self.output_dim,
                      "init": self.init.__name__,
                      "inner_init": self.inner_init.__name__,
                      "forget_bias_init": self.forget_bias_init.__name__,
                      "activation": self.activation.__name__,
                      "inner_activation": self.inner_activation.__name__,
                      "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                      "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                      "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                      "dropout_W": self.dropout_W,
                      "dropout_U": self.dropout_U}
            base_config = super(BiLSTMv1, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


# In[23]:

"""from keras.models import Sequential
from keras.layers import Dense,LSTM
model = Sequential()
model.add(BiLSTMv1(32, input_shape=(10, 64),return_sequences=True))
#model.add(MLSTM(32, input_shape=(10, 32),return_sequences=True))
#model.add(LSTM(32, input_shape=(10, 32),return_sequences=True))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
ins = np.zeros((20,10,64))
model.predict(ins).shape
"""


# In[6]:

import numpy as nps
class BiSimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,close=False, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.close = close
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        super(BiSimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

      
        
        if self.close:
            self.W_h =  self.init((self.output_dim, self.output_dim))
            self.b_h = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W, self.U, self.b]
        
        if self.close:
            self.trainable_weights += [self.W_h,self.b_h]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
    
    def get_output(self,train=False):
        
        self.go_backwards = False
        R1 = Recurrent.get_output(self,train)

        self.go_backwards = True
        R2 = Recurrent.get_output(self,train)

        if self.return_sequences:
            if K._BACKEND == 'tensorflow':
                R2 = tf.reverse(R2,[False,True,False])
            else:
                R2 = R2[::,::-1,::]
        if self.close:
            return  K.dot(R1 + R2 ,self.W_h) + self.b_h
        else:
            return  R1 / 2 + R2 / 2

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        # states contains the previous output,
        # and the two dropout matrices from self.get_constants()
        assert len(states) == 1  # 1 state and 2 constants
        prev_output = states[0]
      
        h = K.dot(x,  self.W) + self.b
        output = self.activation(h + K.dot(prev_output , self.U))
        return output, [output]


    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[9]:

import keras
if  int(keras.__version__.split(".")[0]) >= 1.0 :
    print "v1"
    from keras import activations, initializations, regularizers
    from keras.engine import Layer, InputSpec
    from keras.layers.recurrent import time_distributed_dense,Recurrent
    
    
    class BiSimpleRNNv1(Recurrent):
        '''Fully-connected RNN where the output is to be fed back to input.

        # Arguments
            output_dim: dimension of the internal projections and the final output.
            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the input weights matrices.
            U_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
            dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

        # References
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''
        def __init__(self, output_dim,
                     init='glorot_uniform', inner_init='orthogonal',
                     activation='tanh',close = True,
                     W_regularizer=None, U_regularizer=None, b_regularizer=None,
                     dropout_W=0., dropout_U=0., **kwargs):
            self.output_dim = output_dim
            self.init = initializations.get(init)
            self.inner_init = initializations.get(inner_init)
            self.activation = activations.get(activation)
            self.W_regularizer = regularizers.get(W_regularizer)
            self.U_regularizer = regularizers.get(U_regularizer)
            self.b_regularizer = regularizers.get(b_regularizer)
            self.close = close
            self.dropout_W, self.dropout_U = dropout_W, dropout_U

            if self.dropout_W or self.dropout_U:
                self.uses_learning_phase = True
            super(BiSimpleRNNv1, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            if self.stateful:
                self.reset_states()
            else:
                # initial states: all-zero tensor of shape (output_dim)
                self.states = [None]
            input_dim = input_shape[2]
            self.input_dim = input_dim

            self.W = self.init((input_dim, self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, self.output_dim),
                                     name='{}_U'.format(self.name))
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

            if self.close:
                self.W_h =  self.init((self.output_dim, self.output_dim),
                                      name='{}_W_h'.format(self.name))
                self.b_h = K.zeros((self.output_dim,),
                                   name='{}_b_h'.format(self.name))

            self.regularizers = []
            if self.W_regularizer:
                if self.close:
                    self.W_regularizer.set_param(K.concatenate([self.W,self.W_h]))
                else:
                    self.W_regularizer.set_param(self.W)

                                   
                self.regularizers.append(self.W_regularizer)

            if self.U_regularizer:
                self.U_regularizer.set_param(self.U)
                self.regularizers.append(self.U_regularizer)
            if self.b_regularizer:
                if self.close:
                    self.b_regularizer.set_param(K.concatenate([self.b,self.b_h]))
                else:
                    self.b_regularizer.set_param(self.b)

                self.regularizers.append(self.b_regularizer)

            self.trainable_weights = [self.W, self.U, self.b]
            
            if self.close:
                self.trainable_weights += [self.W_h,self.b_h]

            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights

        def reset_states(self):
            assert self.stateful, 'Layer must be stateful.'
            input_shape = self.input_spec[0].shape
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided (including batch size).')
            if hasattr(self, 'states'):
                K.set_value(self.states[0],
                            np.zeros((input_shape[0], self.output_dim)))
            else:
                self.states = [K.zeros((input_shape[0], self.output_dim))]

        def preprocess_input(self, x):
            if self.consume_less == 'cpu':
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[2]
                timesteps = input_shape[1]
                return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                              input_dim, self.output_dim,
                                              timesteps)
            else:
                return x
            
        def call(self,x,mask=None):

            self.go_backwards = False
            R1 = Recurrent.call(self,x,mask=mask)

            self.go_backwards = True
            R2 = Recurrent.call(self,x,mask=mask)

            if self.return_sequences:
                if K._BACKEND == 'tensorflow':
                    R2 = tf.reverse(R2,[False,True,False])
                else:
                    R2 = R2[::,::-1,::]
            if self.close:
                return  K.dot(R1 + R2 ,self.W_h) + self.b_h
            else:
                return  R1 / 2 + R2 / 2

        def step(self, x, states):
            prev_output = states[0]
            B_U = states[1]
            B_W = states[2]

            if self.consume_less == 'cpu':
                h = x
            else:
                h = K.dot(x * B_W, self.W) + self.b

            output = self.activation(h + K.dot(prev_output * B_U, self.U))
            return output, [output]

        def get_constants(self, x):
            constants = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * self.output_dim, 1)
                B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
                constants.append(B_U)
            else:
                constants.append(K.cast_to_floatx(1.))
            if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[-1]
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)
                B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
                constants.append(B_W)
            else:
                constants.append(K.cast_to_floatx(1.))
            return constants

        def get_config(self):
            config = {'output_dim': self.output_dim,
                      'init': self.init.__name__,
                      'inner_init': self.inner_init.__name__,
                      'activation': self.activation.__name__,
                      'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                      'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                      'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                      'dropout_W': self.dropout_W,
                      'dropout_U': self.dropout_U}
            base_config = super(BiSimpleRNNv1, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


# In[11]:

if __name__ == "__main__":
    from keras.models import Graph
    from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM,GRU
    #from keras.objectives import categorical_crossentropy



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap


    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None,5))
    #graph.add_input(name='input2', input_shape=(None,2))
    inside = 50
    #graph.add_node(BiLSTM(output_dim=inside, activation='sigmoid',input_shape=(200,5),
    #                    inner_activation='hard_sigmoid',return_sequences=True),
    #                   name="l1",input="input1")
    graph.add_node(BiSimpleRNNv1(output_dim=inside, activation='sigmoid',input_shape=(None,5),return_sequences=True),
                       name="l1",input="input1")
    graph.add_output(name="output",input="l1")

    graph.compile('adadelta', {'output':"categorical_crossentropy"})
    print graph.predict({"input1":np.zeros((20,100,5))})["output"].shape


# In[ ]:




# In[ ]:



