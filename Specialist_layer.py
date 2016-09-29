
# coding: utf-8

# In[2]:

from keras.objectives import categorical_crossentropy
import theano.tensor as T
import theano
import numpy as np
#categorical_crossentropy??
#Loss:


perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
perm = [[-3,-2,-1]+iperm for iperm in perm]
perm = np.array(perm,dtype=np.int)
perm += 3
#print perm

test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

eps =1e-7
test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
             [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


def perm_loss(y_true,y_pred):
    def loss(m,  y_true, y_pred,perm):

        #return  perm[T.cast(m,"int32")]
        return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

    #perm = np.array([[0,1],[1,0]],dtype=np.int)
    perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                     [0, 1, 2, 4, 5, 3, 6],
                     [0, 1, 2, 5, 4, 3, 6],
                     [0, 1, 2, 3, 5, 4, 6],
                     [0, 1, 2, 4, 3, 5, 6],
                     [0, 1, 2, 5, 3, 4, 6]],dtype=np.int)
    
    """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                     [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
    seq = T.arange(len(perm))
    result, _ = theano.scan(fn=loss, outputs_info=None, 
                             sequences=seq, non_sequences=[y_true, y_pred,perm])
    return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)

#r =perm_loss(test_true,test_pred).eval()
#print r
#print r.shape


# In[ ]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from keras.objectives import categorical_crossentropy


def reverse(X):
    return X[::,::,::-1]

def output_shape(input_shape):
    # here input_shape includes the samples dimension
    return input_shape  # shap


def sub_mean(X):
    xdms = X.shape
    return X.reshape(xdms[0])

def old_version(ndim=2):

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10
    graph.add_node(Convolution1D(nb_filter=5,filter_length=4,input_shape=(None,5),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4



    graph.add_node(LSTM(output_dim=20, activation='sigmoid',input_shape=(200,10),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="allmost",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")
    graph.add_node(Lambda(reverse, output_shape),inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat",
                   name="reversed0")

    graph.add_node(LSTM(output_dim=20, activation='sigmoid',
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="allmost1",input="reversed0")


    graph.add_node(Lambda(reverse, output_shape),input="allmost1",name="reversed")



    #Here get the subcategory
    graph.add_node(TimeDistributedDense(7,activation="softmax"),inputs=["allmost","reversed"],
                   name="output0",merge_mode="concat",concat_axis=-1)

    ##########################################
    #First ehd here
    #graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    #Here get the number of category
    graph.add_node(LSTM(output_dim=27, activation='softmax',
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0",input="output0")

    graph.add_node(Reshape((1,27)),input="category0",name = "category00")




    graph.add_output(name="output",input="output0")
    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category00")

    graph.compile('adadelta', {'output':'categorical_crossentropy',
                              'category':'categorical_crossentropy' })
    
    return graph



# In[2]:

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

def old_but_ok(ndim=2):
#middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10,filter_length=4,input_shape=(None,5),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation

    inside=50

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid',input_shape=(200,15),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="1allmost",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")



    graph.add_node(Lambda(reverse, output_shape),inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat",
                   name="reversed0")

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid',
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="allmost1",input="reversed0")

    graph.add_node(Lambda(reverse, output_shape),input="allmost1",name="reversed")

    #END first

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid',input_shape=(200,2*inside+15),
                        inner_activation='hard_sigmoid',return_sequences=True),name="allmost_l2",
                       inputs=["input1","input1b","1allmost","reversed"],merge_mode="concat",concat_axis=-1)

    graph.add_node(Lambda(reverse, output_shape),inputs=["input1","input1b","1allmost","reversed"],merge_mode="concat",
                   concat_axis=-1,
                   name="reversed0_l2")

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid',input_shape=(200,2*inside+15),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="allmost1_l2",input="reversed0_l2")

    graph.add_node(Lambda(reverse, output_shape),input="allmost1_l2",name="reversed_l2")

    #END second



    graph.add_node(Dropout(0.4),inputs=["1allmost","reversed","allmost_l2","reversed_l2"],
                   merge_mode="concat",concat_axis=-1,name="output0_drop")

    #Here get the subcategory

    graph.add_node(TimeDistributedDense(7,activation="softmax"),input="output0_drop",
                   name="output0")





    ##########################################
    #First ehd here
    #graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    #Here get the number of category
    graph.add_node(LSTM(output_dim=12,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0_r",input="output0")

    graph.add_node(LSTM(output_dim=12,
                        inner_activation='hard_sigmoid',return_sequences=False,go_backwards=True),
                       name="category0_l",input="output0")

    graph.add_node(Dense(12,activation="softmax"),inputs=["category0_l","category0_r"],concat_axis=1,merge_mode="concat",
                   name="category0")

    graph.add_node(Reshape((1,12)),input="category0",name = "category00")


    #graph.load_weights("step_check")
    #############################################
    #Original end there
    #graph.load_weights("step_check")

    #graph.add_output(name="category",input="category0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy'})

    #############################################

    #graph.add_node(TimeDistributedDense(1,activation="linear"),input='output0',name="output1")


    #graph.load_weights("step_check_bigger")


    graph.add_output(name="output",input="output0")
    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category00")

    graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy'})


    graph.load_weights("old_weights/specialist_4_diff_size_50")
    
    return graph


# In[1]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from Bilayer import BiLSTM
#from keras.objectives import categorical_crossentropy


def return_two_layer():

    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10,filter_length=4,input_shape=(None,5),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation

    inside=50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,15),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="l1",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+15),
                        inner_activation='hard_sigmoid',return_sequences=True),name="l2",
                       inputs=["input1","input1b","l1"],merge_mode="concat",concat_axis=-1)


    graph.add_node(Dropout(0.4),inputs=["l1","l2"],
                   merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(7,activation="softmax"),input="output0_drop",
                   name="output0")




    ##########################################
    #First ehd here
    #graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    #Here get the number of category
    graph.add_node(BiLSTM(output_dim=12,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0bi",input="output0")

    graph.add_node(Dense(12,activation="softmax"),input="category0bi",name="category0")

    graph.add_node(Reshape((1,12)),input="category0",name = "category00")


    #graph.load_weights("step_check")
    #############################################
    #Original end there
    #graph.load_weights("step_check")

    #graph.add_output(name="category",input="category0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy'})

    #############################################

    #graph.add_node(TimeDistributedDense(1,activation="linear"),input='output0',name="output1")


    #graph.load_weights("step_check_bigger")


    graph.add_output(name="output",input="output0")
    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category00")

    graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy'})

    graph.load_weights("saved_weights/two_bilayer_without_sub")
    return graph
    #graph.load_weights("training_general_scale10")
    #############################################
    #Second end there


#############################################



#history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
#predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
#graph.save_weights("step_check",overwrite=True)



# In[6]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from Bilayer import BiLSTM
#from keras.objectives import categorical_crossentropy

def return_three_layer():

    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10,filter_length=4,input_shape=(None,5),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation

    inside=50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,15),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="l1",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+15),
                        inner_activation='hard_sigmoid',return_sequences=True),name="l2",
                       inputs=["input1","input1b","l1"],merge_mode="concat",concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+15),
                        inner_activation='hard_sigmoid',return_sequences=True),name="l3",
                       inputs=["input1","input1b","l2"],merge_mode="concat",concat_axis=-1)


    graph.add_node(Dropout(0.4),inputs=["l1","l2","l3"],
                   merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10,activation="softmax"),input="output0_drop",
                   name="output0")


    graph.add_node(BiLSTM(output_dim=27,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0bi",input="output0")

    graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")

    graph.add_node(Reshape((1,27)),input="category0",name = "category00")



    graph.add_output(name="output",input="output0")
    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category0")

    graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy'})
    
    graph.load_weights("three_layer_specialist")
    return graph

#graph.load_weights("training_general_scale10")
#############################################
#Second end there


#############################################



#history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
#predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
#graph.save_weights("step_check",overwrite=True)



# In[2]:

import theano
#print theano.__version__ , theano.__file__
import keras
#print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from Bilayer import BiLSTM
    
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy





def return_three_bis(ndim=2,inside=50):

    #categorical_crossentropy??
    #Loss:


    perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
    perm = [[-3,-2,-1]+iperm for iperm in perm]
    perm = np.array(perm,dtype=np.int)
    perm += 3

    test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

    eps =1e-7
    test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
                 [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


    def perm_loss(y_true,y_pred):
        def loss(m,  y_true, y_pred,perm):

            #return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7,10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7,10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7,10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7,10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7,10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7,10)],dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None, 
        sequences=seq, non_sequences=[y_true, y_pred,perm])
        return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])
    
    #middle = 50
    add = 0
    if ndim == 3:
        add = 1
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,5+add))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10,filter_length=4,input_shape=(None,5+add),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation


    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,15+add),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="l1",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+15+add),
                        inner_activation='hard_sigmoid',return_sequences=True,),name="l2",
                       inputs=["input1","input1b","l1"],merge_mode="concat",concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+15+add),
                        inner_activation='hard_sigmoid',return_sequences=True),name="l3",
                       inputs=["input1","input1b","l2"],merge_mode="concat",concat_axis=-1)



    graph.add_node(Dropout(0.4),inputs=["l1","l2","l3"],merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10,activation="softmax"),input="output0_drop",
                   name="output0")

    graph.add_node(TimeDistributedDense(4,activation="softmax"),input="output0",
                   name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0bi",input="output0")

    graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")

    graph.add_node(Reshape((1,27)),input="category0",name = "category00")



    graph.add_output(name="output",input="output0")
    graph.add_output(name="outputtype",input="output0b")

    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category0")

    graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy',
                              'outputtype':'categorical_crossentropy'})

    #graph.load_weights("training_general_scale10")
    #############################################
    #Second end there
    
    if ndim == 2 and inside == 50:
        graph.load_weights("saved_weights/three_bilayer_sub_bis")
        
    if ndim == 3 and inside == 50:
        graph.load_weights("saved_weights/three_bilayer_sub_bis_3D_isotrope")

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    #graph.save_weights("step_check",overwrite=True)



# In[9]:

def return_three_bis_three_level(ndim=2,inside=50):

    #categorical_crossentropy??
    #Loss:


    perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
    perm = [[-3,-2,-1]+iperm for iperm in perm]
    perm = np.array(perm,dtype=np.int)
    perm += 3

    test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

    eps =1e-7
    test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
                 [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


    def perm_loss(y_true,y_pred):
        def loss(m,  y_true, y_pred,perm):

            #return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7,10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7,10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7,10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7,10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7,10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7,10)],dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None, 
        sequences=seq, non_sequences=[y_true, y_pred,perm])
        return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])
    
    #middle = 50
    add = 0
    if ndim == 3:
        add = 1
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200,3*(5+add)))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10,filter_length=4,input_shape=(None,3*(5+add)),
                                 border_mode="same"),input='input1',name="conv1")


    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1',name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1',name="input1b")



    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation


    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,10+3*(5+add)),
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="l1",inputs=["input1","input1b"],concat_axis=-1,merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+10+3*(5+add)),
                        inner_activation='hard_sigmoid',return_sequences=True,),name="l2",
                       inputs=["input1","input1b","l1"],merge_mode="concat",concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',input_shape=(200,inside+10+3*(5+add)),
                        inner_activation='hard_sigmoid',return_sequences=True),name="l3",
                       inputs=["input1","input1b","l2"],merge_mode="concat",concat_axis=-1)



    graph.add_node(Dropout(0.4),inputs=["l1","l2","l3"],merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10,activation="softmax"),input="output0_drop",
                   name="output0")

    graph.add_node(TimeDistributedDense(4,activation="softmax"),input="output0",
                   name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0bi",input="output0")

    graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")

    graph.add_node(Reshape((1,27)),input="category0",name = "category00")



    graph.add_output(name="output",input="output0")
    graph.add_output(name="outputtype",input="output0b")

    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category0")

    graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy',
                              'outputtype':'categorical_crossentropy'})

    #graph.load_weights("training_general_scale10")
    #############################################
    #Second end there
    
   
    #############################################

    return graph


# In[12]:

import theano
#print theano.__version__ , theano.__file__
import keras
#print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU

if  int(keras.__version__.split(".")[0]) >= 1.0 :
    from Bilayer import BiLSTMv1 as BiLSTM
else:
    from Bilayer import BiLSTM, BiSimpleRNN    
    
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy





def return_three_bis_simpler(ndim=2,permute=True,extend=0):

    #categorical_crossentropy??
    #Loss:


    perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
    perm = [[-3,-2,-1]+iperm for iperm in perm]
    perm = np.array(perm,dtype=np.int)
    perm += 3

    test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

    eps =1e-7
    test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
                 [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


    def perm_loss(y_true,y_pred):
        def loss(m,  y_true, y_pred,perm):

            #return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7,10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7,10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7,10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7,10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7,10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7,10)],dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None, 
        sequences=seq, non_sequences=[y_true, y_pred,perm])
        return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])
    
    #middle = 50
    add = 0
    if ndim == 3:
        add = 1
   
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None,5+add))
 

    #graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    #66,4


    #First with 20 of activation

    inside=50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                        inner_activation='hard_sigmoid',return_sequences=True),
                       name="l1",input="input1")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                        inner_activation='hard_sigmoid',return_sequences=True,),name="l2",
                       inputs=["input1","l1"],merge_mode="concat",concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                        inner_activation='hard_sigmoid',return_sequences=True),name="l3",
                       inputs=["input1","l2"],merge_mode="concat",concat_axis=-1)



    graph.add_node(Dropout(0.4),inputs=["l1","l2","l3"],merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10 + extend,activation="softmax"),input="output0_drop",
                   name="output0")

    if permute:
        graph.add_node(TimeDistributedDense(4,activation="softmax"),input="output0",
                       name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                        inner_activation='hard_sigmoid',return_sequences=False),
                       name="category0bi",input="output0")

    graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")

    graph.add_node(Reshape((1,27)),input="category0",name = "category00")



    graph.add_output(name="output",input="output0")
    if permute:
        graph.add_output(name="outputtype",input="output0b")

    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category0")

    if permute:
        graph.compile('adadelta', {'output':perm_loss,
                              'category':'categorical_crossentropy',
                              'outputtype':'categorical_crossentropy'})
    else:
        graph.compile('adadelta', {'output':"categorical_crossentropy",
                              'category':'categorical_crossentropy'})

    #graph.load_weights("training_general_scale10")
    #############################################
    #Second end there
   

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    #graph.save_weights("step_check",overwrite=True)



# In[9]:

import theano
#print theano.__version__ , theano.__file__
import keras
#print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
if  int(keras.__version__.split(".")[0]) >= 1.0 :
    from Bilayer import BiLSTMv1 as BiLSTM
else:
    from Bilayer import BiLSTM, BiSimpleRNN    
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy





def return_three_paper(ndim=2,inside=50,permutation=True,inputsize=5,simple=False):

    #categorical_crossentropy??
    #Loss:


    perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
    perm = [[-3,-2,-1]+iperm for iperm in perm]
    perm = np.array(perm,dtype=np.int)
    perm += 3

    test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

    eps =1e-7
    test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
                 [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


    def perm_loss(y_true,y_pred):
        def loss(m,  y_true, y_pred,perm):

            #return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7,10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7,10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7,10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7,10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7,10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7,10)],dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None, 
        sequences=seq, non_sequences=[y_true, y_pred,perm])
        return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])
    
    #middle = 50
    add = 0
    
    if ndim == 3:
        add = 1
        
    if simple:
        Bi = BiSimpleRNN
        
    else:
        Bi = BiLSTM
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None,inputsize))
 

    graph.add_node(Bi(output_dim=inside,activation='tanh',return_sequences=True,close=True,input_shape=(200,inputsize),),
                       name="l1",input="input1")

    graph.add_node(Bi(output_dim=inside,input_shape=(200,inputsize),
                       return_sequences=True,close=True,activation='tanh'),name="l2",
                       inputs=["input1","l1"],merge_mode="concat",concat_axis=-1)

    graph.add_node(Bi(output_dim=inside,activation='tanh', input_shape=(200,inputsize),
                        return_sequences=True,close=True),name="l3",
                       inputs=["input1","l2"],merge_mode="concat",concat_axis=-1)



    graph.add_node(Dropout(0.4),inputs=["l1","l2","l3"],merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10,activation="softmax"),input="output0_drop",
                   name="output0")


    graph.add_node(Bi(output_dim=27,activation='tanh',return_sequences=False,close=True),
                       name="category0bi",input="output0")

    graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")


    graph.add_output(name="output",input="output0")

    #graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category",input="category0")

    if permutation:
        graph.compile('adadelta', {'output':perm_loss,
                                  'category':'categorical_crossentropy'})
    else:
        graph.compile('adadelta', {'output':'categorical_crossentropy',
                                  'category':'categorical_crossentropy'})
        

    #graph.load_weights("training_general_scale10")
    #############################################
    #Second end there

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    #graph.save_weights("step_check",overwrite=True)



# In[22]:

import theano
#print theano.__version__ , theano.__file__
import keras
#print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Merge,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D,MaxPooling1D,UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
if  int(keras.__version__.split(".")[0]) >= 1.0 :
    from Bilayer import BiLSTMv1 as BiLSTM
    from Bilayer import BiSimpleRNNv1 as BiSimpleRNN

else:
    from Bilayer import BiLSTM, BiSimpleRNN
    
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy





def return_layer_paper(ndim=2,inside=50,permutation=True,inputsize=5,simple=False,
                       n_layers=4,category=True,output=True):

    #categorical_crossentropy??
    #Loss:


    perm =[[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]
    perm = [[-3,-2,-1]+iperm for iperm in perm]
    perm = np.array(perm,dtype=np.int)
    perm += 3

    test_true = [[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]],[[0,1],[0,1],[0,1]]]

    eps =1e-7
    test_pred = [[[1-eps,+eps],[1-eps,eps],[1-eps,eps]],[[eps,1-eps],[eps,1-eps],[eps,1-eps]],
                 [[eps,1-eps],[eps,1-eps],[eps,1-eps]]]


    def perm_loss(y_true,y_pred):
        def loss(m,  y_true, y_pred,perm):

            #return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean( T.sum(y_true[::,::,perm[m]] * T.log(y_pred),axis=-1),axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7,10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7,10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7,10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7,10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7,10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7,10)],dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None, 
        sequences=seq, non_sequences=[y_true, y_pred,perm])
        return -T.mean(T.max(result,axis=0)) #T.max(result.dimshuffle(1,2,0),axis=-1)



    def reverse(X):
        return X[::,::,::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])
    
    #middle = 50
    add = 0
    
    if ndim == 3:
        add = 1
        
    if simple:
        Bi = BiSimpleRNN
        
    else:
        Bi = BiLSTM
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None,inputsize))
 

    graph.add_node(Bi(output_dim=inside,activation='tanh',return_sequences=True,close=True,input_shape=(200,inputsize),),
                       name="l1",input="input1")
    
    
    for layer in range(2,n_layers+1):

        graph.add_node(Bi(output_dim=inside,input_shape=(200,inputsize),
                           return_sequences=True,close=True,activation='tanh'),name="l%i"%layer,
                           inputs=["input1","l%i"%(layer-1)],merge_mode="concat",concat_axis=-1)


    graph.add_node(Dropout(0.4),inputs=["l%i"%layer for layer in range(1,n_layers+1)],
                   merge_mode="concat",concat_axis=-1,name="output0_drop")
    #Here get the subcategory

    graph.add_node(TimeDistributedDense(10,activation="softmax"),input="output0_drop",
                   name="output0")


    
    res = {}

    if category:
        graph.add_node(Bi(output_dim=27,activation='tanh',return_sequences=False,close=True),
                           name="category0bi",input="output0")
        graph.add_node(Dense(27,activation="softmax"),input="category0bi",name="category0")
        graph.add_output(name="category",input="category0")
        res['category'] = 'categorical_crossentropy'
    
    if output:
        graph.add_output(name="output",input="output0")

        if permutation:
            res['output'] = perm_loss
        else:
            res['output'] = 'categorical_crossentropy'
        

    graph.compile('adadelta', res)
        

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    #graph.save_weights("step_check",overwrite=True)



# In[18]:



