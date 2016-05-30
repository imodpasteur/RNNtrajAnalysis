
# coding: utf-8

# In[2]:

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import newaxis
from numpy import histogram
import copy
#from pylab import plot
from scipy.io import loadmat


# In[6]:

from numpy import mean,cov,cumsum,dot,linalg,array,rank
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm

def M(axis, theta):
    return expm3(cross(eye(3), axis/norm(axis)*theta))


def plot_traj(X,label=[],random_sin=[],toplot=True):
    colors = {0:"b",1:"g",2:"r",3:"k"}

    for i,(d,t) in enumerate(X):
       
        plot(X[i:i+2,0],X[i:i+2:,1],color=colors[label[i]])
        
def random_rot(traj,alpha=None,ndim=2,axis=[]):
    
    if ndim == 2:
        if alpha is None:
            alpha = 2*3.14*np.random.random()
        if axis == []:
            axis = [[np.cos(-alpha),np.cos(-alpha+3.14/2)],
                   [np.sin(-alpha),np.sin(-alpha+3.14/2)]]

        axis=np.array(axis)
        
    if ndim == 3:
        if alpha is None:
            alpha = 3.14*np.random.random()
        if axis == []:
            axis = np.random.random(3)
            
        axis = M(axis,alpha)
        
        #print axis
    
    #print axis.shape
    newtraj =  (traj-mean(traj.T,axis=1)).T 

    return dot(axis.T,newtraj).T

#print random_rot(np.zeros((10,3)),0,ndim=3)


# In[91]:

colors = {0:"b",1:"g",2:"r",3:"k",4:"y",5:"c",6:"m"}
import pylab as plt
import colorsys
import numpy as np
N = 15
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

colors= dict(colors.items() + {i+7:c for i,c in enumerate( RGB_tuples[::3]) }.items())

#print colors
def plot_label(traj,seq,remove6=None,linewidth=3,markersize=5,k=None):
    #colors
    #colors = {0:"b",1:"g",2:"r",3:"k",4:"y",5:"c",6:"m",7:"b",8:"g",9:"r"}
    #print traj.shape,seq.shap
    init_seq = seq[0]
    i = 0 
    start = 0
    while i < len(traj)-1:
        start = i
        init_seq = seq[i]
        while i < len(traj)-1 and seq[i] == init_seq:
            i += 1
        #print start,i,traj[start:i+1,0]
        #print 
        if remove6:
            if init_seq  == remove6:
                continue
        if k is None:
            plt.plot(traj[start:i+1,0],traj[start:i+1,1],"-o",color=colors[init_seq],
                 linewidth=linewidth,markersize=markersize)
        if k in [0,1]:
            plt.plot(range(start,i+1),traj[start:i+1,k],"-o",color=colors[init_seq],
                 linewidth=linewidth,markersize=markersize)


# In[353]:

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,labels=[],rotation=45):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    #print len(labels),cm.shape[0]

    assert len(labels) == cm.shape[0]
    plt.xticks(tick_marks, labels, rotation=rotation)
    plt.yticks(tick_marks, labels,rotation=0)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[742]:

from sklearn.metrics import confusion_matrix

def get_statistiques_hmm(Y_tests,Y_test_cat,root="",zone=range(500)):
    hmm = []
    SVM = []
    Class_HMM = []
    

    mv = []
    mv_HMM = []
    for i in zone:

        PrM,states,labels,possible,possible2,t = get_step_class(root+"/res%i.mat"%i)

        init = np.argmax(Y_tests[i],axis=1)

        l=len(Y_tests[i])-1
        #print l

        if possible != []:
            #print init
            #print possible
            diff = [ np.sum(init[:l] != possible[k][:l]) for k in range(len(possible))]
            #print diff
            states = possible[np.argmin(diff)]
            #print i,diff
            #print i,len(possible),labels
            """ 
            diff = [ np.sum(init[:l] != possible2[k][:l]) for k in range(12)]
            states = possible2[np.argmin(diff)]
            """



        Class_HMM.append(states[:l])
        hmm.append(np.sum(init[:l] != states[:l])*100./l)

        mv.extend(init[:l])
        mv_HMM.extend(states[:l])

    hmm = np.array(hmm)
  

    mvc = []
    mvc_HMM = []


    HMM = []
    GT = []
    misp_HMM = []
    correct_HMM = []

    for i in zone:

        PrM,_,labels,possible,possible2,_ = get_step_class(root+"/res%i.mat"%i)

        cat = np.argmax(Y_test_cat[i])

        catHMM = np.argmax(PrM)

        HMM.append(catHMM)
        GT.append(cat)

        mvc.append(cat)
        mvc_HMM.append(catHMM)


        if cat != catHMM:
            misp_HMM.append(PrM[cat])
        else:
            correct_HMM.append(PrM[cat])

    hmm = np.array(hmm)

    GT = np.array(GT)

    return [np.sum(np.array(mv) != np.array(mv_HMM)) *100. / len(mv), len(mv) , confusion_matrix(mv,mv_HMM),             np.sum(GT != mvc_HMM) *100./len(GT) ,len(GT), confusion_matrix(mvc,mvc_HMM)]


# In[748]:

from sklearn.metrics import confusion_matrix


def get_statistiques(Y_tests,Y_test_cat,pred_RNN,pred_RNN_cat,fight=False,sub=True):

    RNN = []

    Class_RNN = []

    zone = range(0,len(pred_RNN))
    mv = []
    mv_RNN = []

    for i in zone:



        init = np.argmax(Y_tests[i],axis=1)

        l=len(Y_tests[i])-1
        #print l


        classi_RNN = np.argmax(pred_RNN[i][:l],axis=1)


        delta_RNN = np.sum(classi_RNN != init[:l]) 
        
        perm = np.array([[0, 1, 2, 3, 4, 5, 6],
             [0, 1, 2, 4, 5, 3, 6],
             [0, 1, 2, 5, 4, 3, 6],
             [0, 1, 2, 3, 5, 4, 6],
             [0, 1, 2, 4, 3, 5, 6],
             [0, 1, 2, 5, 3, 4, 6]],dtype=np.int)

        if sub:
            perm = np.array([[0, 1, 2, 3, 4, 5, 6]+range(7,10),
                         [0, 1, 2, 4, 5, 3, 6]+range(7,10),
                         [0, 1, 2, 5, 4, 3, 6]+range(7,10),
                         [0, 1, 2, 3, 5, 4, 6]+range(7,10),
                         [0, 1, 2, 4, 3, 5, 6]+range(7,10),
                         [0, 1, 2, 5, 3, 4, 6]+range(7,10)],dtype=np.int)
        deltas = []
        for permutation in perm:
            classi_RNN = clean(pred_RNN[i][:l,permutation],np.argmax(pred_RNN_cat[i]),fight=fight,sub=sub)

            deltas.append(np.sum(classi_RNN != init[:l]) )

        best = np.argmin(deltas)
        classi_RNN = clean(pred_RNN[i][:l,perm[best]],np.argmax(pred_RNN_cat[i]),fight=fight,sub=sub)

        Class_RNN.append(classi_RNN)
        RNN.append(np.sum(init[:l] != classi_RNN[:l])*100./l)

        mv.extend(init[:l])
        mv_RNN.extend(classi_RNN[:l])

    RNN = np.array(RNN)

    mvc = []
    mvc_RNN = []


    RNN = []
    GT = []
    for i in zone:


        cat = np.argmax(Y_test_cat[i])

        catRNN = np.argmax(pred_RNN_cat[i])


        RNN.append(catRNN)
        GT.append(cat)

        mvc.append(cat)
        mvc_RNN.append(catRNN)

    RNN = np.array(RNN)
    GT = np.array(GT)

    #print range(pred_RNN_cat[0].shape[1])
    return np.sum(np.array(mv) != np.array(mv_RNN)) *100. / len(mv), len(mv) , confusion_matrix(mv,mv_RNN),             np.sum(GT != RNN) *100./len(RNN) ,len(RNN), confusion_matrix(mvc,mvc_RNN,labels=range(pred_RNN_cat[0].shape[1]))


# In[7]:

from prePostTools import traj_to_dist,get_parameters


# In[8]:

#colors = {0:"b",1:"g",2:"r",3:"k",4:"y",5:"c",6:"m"}

def plot_by_class(traj,seq):
    #global colors
    #colors = {0:"b",1:"g",2:"r",3:"k",4:"y",5:"c",6:"m"}
    #print traj.shape,seq.shap
    init_seq = seq[0]
    i = 0 
    start = 0
    
    delta = traj[1:]-traj[:-1]
    
    for i in range(9):
        c = np.array(seq) == i
        plt.plot(delta[c,0],delta[c,1],"o",color=colors[i])


# In[1]:

Labels = ['D','DV','D, D','D, DV','DV, DV','D, D, D','D, D, DV','D, DV, DV','DV, DV, DV']

from prePostTools import M1,M0

#print len(M0),len(M1)


# In[10]:

def get_param(fich="/home/jarbona/RNN_mus/res0",first=False):
    
    left=3
    right=4
    middle=5
 

    Mp = loadmat(fich,squeeze_me=False)
    kPrM=0
    kML_states= 1
    kML_params = 2
    ktrack = 5
    result = Mp["results"][0][0]
    PrM = result[kPrM][0]
    ML_params = result[kML_params][0]
    
    track = result[ktrack][0]
    ML_states = result[kML_states][0]
    
    #print len(track),len(ML_states)
    
    sigmas = ML_params[0][-1][0]
    emit = ML_params[0][-2]
    return ML_params


# In[11]:

def get_step_class(fich="/home/jarbona/RNN_mus/res0",first=False):
    
    left=3
    right=4
    middle=5
 

    Mp = loadmat(fich,squeeze_me=False)
    kPrM=0
    kML_states= 1
    kML_params = 2
    ktrack = 5
    result = Mp["results"][0][0]
    PrM = result[kPrM][0]
    ML_params = result[kML_params][0]
    
    track = result[ktrack][0]
    ML_states = result[kML_states][0]
    
    #print len(track),len(ML_states)
    
    sigmas = ML_params[0][-1][0]
    emit = ML_params[0][-2]
    #print emit
    possible = []
    possible2 = []
    
    states = np.array(ML_states,dtype=np.int32)-1
    
    classt= Labels[np.argmax(PrM)]
    
    if classt == 'D, D, D':
        translate = [[sigmas[i],i] for i in range(3)]
        translate.sort()
        translate = { index[1]+3:i for i,index in enumerate(translate)}
        states += 3
        for k,v in translate.items():
            states[states == k] = v
        
    if classt == 'DV':
        
        states[states == 0] = left
    
    if classt == 'D, D':
        if sigmas[0] > sigmas[1]:
            states = -(states-1)
            
    if classt == 'DV, DV':
      
            
        possible = [states.copy() for i in range(2)]

        possible[0][states == 0] = left
        possible[0][states == 1] = right
        n=1
        possible[1][states == 0] = right
        possible[1][states == 1] = left
    
    if classt == 'D, D, DV':
        #print "sigma",sigmas
        #print "emit",emit
        if sigmas[0] > sigmas[1]:
            cp = states.copy()
            states[cp==0] = 1
            states[cp==1] = 0
            
      
        possible = [states.copy() for i in range(3)]
        

        possible[0][states == 2] = left
    
        possible[1][states == 2] = right
    
        possible[2][states == 2] = middle
        
        states[ states == 2] = left

        
    if classt == 'D, DV':
        
       
        states[states==1] = left


            
    if classt == 'D, DV, DV':
        
        possible = [states.copy() for i in range(2)]

        possible[0][states == 1] = left
        possible[0][states == 2] = right
        n=1
        possible[1][states == 1] = right
        possible[1][states == 2] = left
        
        
        states[ states == 1] = left
        states[ states == 2] = right
        
        

    if classt == 'DV, DV, DV':
        #print emit
        #possible = [states.copy() for i in range(6)]
        possible2 = [states.copy() for i in range(6)]

        n = 0
        m= 0 
        for i,j,o in [[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,0,1]]:
            
            #possible[n][states == i] = left
            #possible[n][states == j] = right
            #n += 1
            #possible[n][states == i] = right
            #possible[n][states == j] = left
            #n += 1
            
            possible2[m][states == i] = left
            possible2[m][states == j] = right
            possible2[m][states == o] = middle
            m += 1
           
        possible = possible2
    
    return PrM,states,Labels[np.argmax(PrM)],possible,possible2,track


# In[12]:




# In[13]:

def generate_alternative(x):
    x = np.array(x)
    x = np.array([x.copy() for i in range(6)])
    
    if 3 in x and 4 in  x:
        x[1][x[0] == 3] == 4
        x[1][x[0] == 4] == 3
    if 3 in x and 4 in  x and 5 in x:
        n = 0
        for i,j,o in [[0,1,2],[1,2,0],[2,1,0],[0,2,1],[1,0,2],[2,1,0]]:
            x[n][x[0] == 3] == i+3
            x[n][x[0] == 4] == j+3
            x[n][x[0] == 5] == o+3
            n += 1
    return x


# In[4]:

from prePostTools import clean
#import yahmm
#yahmm.DiscreteDistribution??


# In[7]:

import copy
def create_random_alpha(N,alpha=0.5,drift=[0,0,0],ndim=2):
    coord = []
    hurstExponent = alpha / 2.

    L = int(N/128.)+1


    scaleFactor = 2 ** (2.0 * hurstExponent)

    def curve2D(t0, x0,y0, t1, x1,y1, variance, scaleFactor):
        if (t1 - t0) < .01:
            #print 
            #stddraw.line(x0, y0, x1, y1)
            coord.append([t1,x1,y1])
            return #[x0, y0] ,[x1, y1]
        tm = (t0 + t1) / 2.0
        ym = (y0 + y1) / 2.0
        xm = (x0 + x1) / 2.0
        deltax = np.random.normal(0, math.sqrt(variance))
        deltay = np.random.normal(0, math.sqrt(variance))

        curve2D(t0,x0, y0, tm, xm+deltax,ym+deltay, variance/scaleFactor, scaleFactor)
        curve2D(tm,xm+deltax, ym+deltay, t1,x1, y1, variance/scaleFactor, scaleFactor)

        
    def curve3D(t0, x0,y0,z0 ,t1, x1,y1,z1, variance, scaleFactor):
        if (t1 - t0) < .01:
            #print 
            #stddraw.line(x0, y0, x1, y1)
            coord.append([t1,x1,y1,z1])
            return #[x0, y0] ,[x1, y1]
        tm = (t0 + t1) / 2.0
        ym = (y0 + y1) / 2.0
        xm = (x0 + x1) / 2.0
        zm = (z0 + z1) / 2.0

        deltax = np.random.normal(0, math.sqrt(variance))
        deltay = np.random.normal(0, math.sqrt(variance))
        deltaz = np.random.normal(0, math.sqrt(variance))

        curve3D(t0,x0, y0,z0 ,tm, xm+deltax,ym+deltay,zm+deltaz, variance/scaleFactor, scaleFactor)
        curve3D(tm,xm+deltax, ym+deltay,zm+deltaz, t1,x1, y1,z1, variance/scaleFactor, scaleFactor)


    scale_step = 8.5
    if ndim == 2:
        curve2D(0., 0.,0., L,0.+drift[0], 0.0+drift[1], scale_step, scaleFactor)
    if ndim == 3:
        curve3D(0., 0.,0.,0, L,0.+drift[0], 0.0+drift[1], 0.0+drift[2], scale_step, scaleFactor)

    #print L
    return np.array(coord)[::,1:]

if __name__ == "__main__":
    coord = []
    
    def curve(t0, y0, t1, y1, variance, scaleFactor):
        if (t1 - t0) < .01:
            #print 
            #stddraw.line(x0, y0, x1, y1)
            coord.append([t1,y1])
            return #[x0, y0] ,[x1, y1]
        tm = (t0 + t1) / 2.0
        ym = (y0 + y1) / 2.0
        delta = np.random.normal(0, math.sqrt(variance))
        curve(t0, y0, tm, ym+delta, variance/scaleFactor, scaleFactor)
        curve(tm, ym+delta, t1, y1, variance/scaleFactor, scaleFactor)
        

    #-----------------------------------------------------------------------

    # Accept a Hurst exponent as a command-line argument.
    # Use the Hurst exponent to compute a scale factor.
    # Draw a Brownian bridge from (0, .5) to (1.0, .5) with
    # variance .01 and that scale factor.

   
    #curve(0, .5, 25, .5, .01, scaleFactor)
    #coord = np.array(coord)
    #plot(coord[::,0],coord[::,1])
    
    coord = np.array(create_random_alpha(200.,alpha=0.5))
    print coord.shape

    plot(coord[:200,0],coord[:200,1])
    #plot(coord[::,1])
    print (coord[1:,1]-coord[:-1,1]).std()
    
    
    


# In[376]:

import random
import math
import copy
import types
from yahmm import *

#from matplotlib.axis import axis
"""
print [ k for k in lic2[ch-1].keys() if "x" in k]
with open("/home/jarbona/Theano/subd","w" ) as f:
    gene = "MeancentroMid"
    x,y,z = lic2[ch-1]["x"+gene],lic2[ch-1]["y"+gene],lic2[ch-1]["z"+gene]
    print len(x)
    cPickle.dump([x,y,z],f)

"""


def NormalDistribution(number,pre):
    return DiscreteDistribution({number:1})
def one_particle_n_states(ListState0,transition_mat=[],StateN={},selfprob=0.033):
  

    model = Model( name="Unknown" )
    
    pre=0.0001
    Ra0 = State(NormalDistribution(StateN["Ra0"], pre), name="Ra0")
    Ra1 = State(NormalDistribution(StateN["Ra1"], pre), name="Ra1") 
     
    Ra2 = State(NormalDistribution(StateN["Ra2"], pre), name="Ra2")   
    
    
    sRa0 = State(NormalDistribution(StateN["sRa0"], pre), name="sRa0")
    sRa1 = State(NormalDistribution(StateN["sRa1"], pre), name="sRa1") 
     
    sRa2 = State(NormalDistribution(StateN["sRa2"], pre), name="sRa2")   
    """
    Ra3 = State(NormalDistribution(StateN["Ra3"], pre), name="Ra3")
    Ra4 = State(NormalDistribution(StateN["Ra4"], pre), name="Ra4")   
    Ra5 = State(NormalDistribution(StateN["Ra5"], pre), name="Ra5") 
    
    """
    Le0 = State(NormalDistribution(StateN["Le0"], pre), name="Le0")
    #Le1 = State(NormalDistribution(StateN["Le1"], pre), name="Le1")
    #i0 = State(NormalDistribution(StateN["Ri0"], pre), name="Ri0")
    Ri0 = State(NormalDistribution(StateN["Ri0"], pre), name="Ri0")

    Ri1 = State(NormalDistribution(StateN["Ri1"], pre), name="Ri1")

    
    ListStatet=[Ra0,Ra1,Ra2,Le0,Ri0,Ri1,sRa0,sRa1,sRa2]#,Ra3,Ra4,Ra5,Le0]#,Ri0]#,Ri1,Le1]
    ListState = []
    for i in ListState0:
        ListState.append(ListStatet[i])
        
    #print [l.name for l in ListState]
    for state in ListState:
        model.add_state(state)
       
    endp=0.0000001
    for state0 in ListState:
        for state1 in ListState:
            if state1.name == state0.name:
                s0=0.166
                s0=selfprob
                model.add_transition(state0, state1,s0)
            else:
                model.add_transition(state0,state1, (1-s0-endp)/(len(ListState)-1) )

   
    for state in ListState:

        model.add_transition(model.start,state, 1.0/len(ListState))
        model.add_transition(state,model.end, endp)
   

    model.bake()
    
    return model



def generate_traj(time,fight=False,diff_sigma=2,deltav=0.4,
                  delta_sigma_directed=6,force_model=None,
                  lower_selfprob=0.4,zeros=True,Ra0 = [],Ra1 = [],Mu0=[],Mu1=[],sRa0=[],
                  sub=False,clean=None,check_delta=False,alpha=0.5,ndim=2,anisentropy=0.5,rho_fixed=False):

    global X,Y
    #nstate = np.random.randint(1,6)
    #nstate=6
    #ListState = range(6)
    #np.random.shuffle(ListState)
    #ListState = ListState[:nstate]
    
    
    
    #0 1 = random
    #2 3 = Left
    #4 5 = Right
    
    #if 1 in ListState and not 0 in ListState:
    #    ListState[ListState.index(1)] = 0
    #if 3 in ListState and not 2 in ListState:
    #    ListState[ListState.index(3)] = 2
    #if 5 in ListState and not 4 in ListState:
    #    ListState[ListState.index(5)] = 4

    StateN = {"Ra0": 0,"Ra1":1,"Ra2":2,"Le0":3,"Ri0":4,"Ri1":5,"sRa0":6,"sRa1":7,"sRa2":8}#,"Ra3": 3,"Ra4":4,"Ra5":5,"Le0":6}
    iStateN = {v:k for k,v in StateN.items()}
    
    Model_type = {"D":[0,["Ra0"]],
                  "Dv":[1,["Le0"]],
                  "D-D":[2,["Ra0","Ra1"]],
                  "D-DvL":[3,["Ra0","Le0"]],
                  "DvR-DvL":[4,["Le0","Ri0"]],
                  "D-D-D":[5,["Ra0","Ra1","Ra2"]],
                  "D-D-DvL":[6,["Ra0","Ra1","Le0"]],
                  "D-DvL-DvR":[7,["Ra0","Le0","Ri0"]],
                  "D-D-DvL-DvR":[8,["Ra0","Ra1","Le0","Ri0"]],  
                  "DvL-DvR-DvR1":[9,["Le0","Ri0","Ri1"]],
                  "D-DvL-DvR-DvR1":[10,["Ra0","Le0","Ri0","Ri1"]],
                  "D-D-DvL-DvR-DvR1":[11,["Ra0","Ra1","Le0","Ri0","Ri1"]]}
    
    if sub:
        Model_type1 = {"sD":[12,["sRa0"]],
                      "D-sD":[13,["Ra0","sRa0"]],
                      "sD-sD":[14,["sRa0","sRa1"]],
                      "sD-DvL":[15,["sRa0","Le0"]],                     
                      "sD-D-D":[16,["sRa0","Ra0","Ra1"]],
                      "sD-sD-D":[17,["sRa0","sRa1","Ra0"]],
                      "sD-sD-sD":[18,["sRa0","sRa1","sRa2"]],      
                      "sD-D-DvL":[19,["sRa0","Ra0","Le0"]],
                      "sD-sD-DvL":[20,["sRa0","sRa1","Le0"]],              
                      "sD-DvL-DvR":[21,["sRa0","Le0","Ri0"]],
                      "sD-D-DvL-DvR":[22,["sRa0","Ra0","Le0","Ri0"]],  
                      "sD-sD-DvL-DvR":[23,["sRa0","sRa1","Le0","Ri0"]],                 
                      "sD-DvL-DvR-DvR1":[24,["sRa0","Le0","Ri0","Ri1"]],
                      "sD-D-DvL-DvR-DvR1":[25,["sRa0","Ra0","Le0","Ri0","Ri1"]],
                      "sD-sD-DvL-DvR-DvR1":[26,["sRa0","sRa1","Le0","Ri0","Ri1"]]
                     }
        Model_type = dict(Model_type.items()+Model_type1.items())
        
                  #}
    if fight:
        Model_type.pop("D-D-DvL-DvR")
        Model_type.pop("D-DvL-DvR-DvR1")
        Model_type.pop("D-D-DvL-DvR-DvR1")
        Model_type["DvL-DvR-DvR1"][0] = 8
                
    """
    
    Model_type = {"D":[0,["Ra0"]],"D-D":[1,["Ra0","Ra1"]],"D-D-D":[2,["Ra0","Ra1","Ra2"]]}
    #print ListState
    Model_num =  np.random.randint(0,3)
    ModelN = Model_type.keys()
    ModelN.sort()
    ModelN = ModelN[Model_num]
    Model_num=Model_type[ModelN][0]"""
    
    iModel = {v[0]:k for k,v in Model_type.items()}
    Model_num =  np.random.randint(0,len(Model_type.keys()))
    
    if force_model is not None:
        if type(force_model) == types.IntType:
            Model_num = force_model
        else:
            Model_num =  force_model(np.random.randint(0,len(force_model)))

    #Model_num = 9
    
    
    
    ListState = [StateN[iname] for iname in Model_type[iModel[Model_num]][1]]

    #ListState=[0]
    selfprob = lower_selfprob + (1-lower_selfprob)*random.random()
    #print "Sampling",ListState
    model = one_particle_n_states(ListState0=ListState,StateN=StateN,selfprob=selfprob)
    #print "ENdS"
    seq = np.zeros(time)
    sequence  = model.sample(time)
    
    
    scale = 1+9*random.random()
    
    
    
    cats = scale*np.random.random()
   
    diff_sigma=2
    #1.5
    
    if Ra0 == []:
        
        Ra0 = [0,cats]
    else:
        scale = 1
        
    
    
    if Ra1 == []:
        
        Ra1 = [0,max(diff_sigma*Ra0[1]+scale*np.random.random(),scale)]
    else:
        scale = 1
    
    
    Ra2 = [0,max(diff_sigma*Ra1[1]+scale*np.random.random(),scale)]
    
    R_anisentropy = {"Ra0":1-anisentropy +  anisentropy*(1-2*np.random.random(3)),
                   "Ra1":1-anisentropy +  anisentropy*(1-2*np.random.random(3)),
                   "Ra2":1-anisentropy +  anisentropy*(1-2*np.random.random(3))}
    
    D ={"Ra0": Ra0,"Ra1":Ra1,"Ra2":Ra2} 
    
    ##############################################
    #Sub
    cats = scale*np.random.random()

    if sRa0 == []:
        
        sRa0 = [0,cats]
    else:
        scale = 1
    sRa1 = [0,max(diff_sigma*sRa0[1]+scale*np.random.random(),scale)]
    
    
    sRa2 = [0,max(diff_sigma*sRa1[1]+scale*np.random.random(),scale)]
    
    sD ={"sRa0": sRa0,"sRa1":sRa1,"sRa2":sRa2} 
    sR_anisentropy = {"sRa0":1-anisentropy +  anisentropy*(1-2*np.random.random(3)),
                      "sRa1":1-anisentropy +  anisentropy*(1-2*np.random.random(3)),
                      "sRa2":1-anisentropy +  anisentropy*(1-2*np.random.random(3))}
   
    start={"sRa0":0,"sRa1":0,"sRa2":0}
    
    #dsub = 0.15*(1-2*np.random.rand(3))
    dsub=[0,0,0]
    sds = {iname:create_random_alpha(time+1,alpha=alpha,ndim=ndim)
           for iname in Model_type[iModel[Model_num]][1] if iname in ["sRa0","sRa1","sRa2"]}
    
    sds = {k:v[1:]-v[:-1] for k,v in sds.items()}
    
    ################################################################
    #Directed

    namesl = []
    alpha2 = 0.15*3.14 + 1.85*3.14*random.random()
    #alpha2 = -3.14
    dalpha2 = max(0.1,0.8*np.random.random())
    dalpha1 = max(0.1,0.8*np.random.random())


    traj = np.zeros((time,ndim))
    tot0 = 0
    tot1 = 0
    
    mus1 = scale*(1-2*np.random.random(ndim))
    
    mus2 = mus1.copy() 
    
    mus3 = mus1.copy() 
    #deltav = 0.1
    while np.sqrt(np.sum((mus1-mus2)**2)) < deltav*scale and             np.sqrt(np.sum((mus1-mus3)**2)) < deltav*scale  and             np.sqrt(np.sum((mus2-mus3)**2)) < deltav*scale :
        
        
        mus1 = 2*scale*(1-2*np.random.random(ndim))
        mus2 = 2*scale*(1-2*np.random.random(ndim))
        mus3 = 2*scale*(1-2*np.random.random(ndim))

        """
        
        if mus1[0] < mus2[0]:
            mus1,mus2=mus2,mus1
            
        if mus1[0] < mus3[0]:
            mus1,mus3=mus3,mus1
            
        if mus2[0] < mus3[0]:
            mus2,mus3=mus3,mus2
        mus1[1] = 0 
        """

    epsilon=1e-7

    rho1 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas1 = scale*np.random.random(ndim) + epsilon
    
    rho2 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas2 = scale*np.random.random(ndim) + epsilon
    
    rho3 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas3 = scale*np.random.random(ndim) + epsilon
    #if np.sum(mus1**2) > np.sum(mus2**2):
    #    mus1,mus2=mus2,mus1
        

    if rho_fixed:
        rho1 = [0,0]
        rho2 = [0,0]
        rho3 = [0,0]
        
    #Before 4
    
    #Borned:
    
    d1 = np.sqrt(np.sum(mus1**2))+epsilon
    """
    if d1 < 0.01:
        mus1 = 0.01*mus1/d1
    """
    for indim in range(ndim):
        sigmas1[indim]  = min(d1/delta_sigma_directed,sigmas1[indim])
    
    
    d2 = np.sqrt(np.sum(mus2**2))+epsilon
    """
    if d2 < 0.01:
        mus2 = 0.01*mus2/d2
    """
    for indim in range(ndim):
        sigmas2[indim]  = min(d2/delta_sigma_directed,sigmas2[indim])
    
    
    d3 = np.sqrt(np.sum(mus3**2))+epsilon
    """
    if d3 < 0.01:
        mus3 = 0.01*mus3/d3
    """
    for indim in range(ndim):
        sigmas3[indim]  = min(d3/delta_sigma_directed,sigmas3[indim])
        
    if Mu0 != []:
        mus1 = Mu0[0]
        sigmas1 = Mu0[1]
        rho1 = Mu0[2]
        
    if Mu1 != []:
        mus2 = Mu1[0]
        sigmas2 = Mu1[1]
        rho2 = Mu1[2]
    #mus2 = scale*
    
    

    if ndim == 2:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf],
                           [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf]]
        
    if ndim == 3:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[0]*sigmasm[2]*rhom[1] * scf],
                    [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf,  sigmasm[1]*sigmasm[2]*rhom[2] * scf],
                    [sigmasm[0]*sigmasm[2]*rhom[1] * scf, sigmasm[1]*sigmasm[2]*rhom[2] * scf, sigmasm[2] * scf]]
        
        
    #print mus1
    SigmasMu = {'Le0':[mus1,get_covariance(sigmas1,rho1)],
              "Ri0":[mus2,get_covariance(sigmas2,rho2)],
              "Ri1":[mus3,get_covariance(sigmas3,rho3)]}
    StartMu = {"Le0":0,"Ri0":0,"Ri1":0}
    
    Mus = { iname:numpy.random.multivariate_normal(SigmasMu[iname][0],SigmasMu[iname][1],time)                    for iname in Model_type[iModel[Model_num]][1] if iname in ["Le0","Ri0","Ri1"]}
    #MODIF
   
    Dim = 2
    for tt,v in enumerate(sequence):
   
        
        
        seq[tt] = int(round(v,0))
        
        
        name = iStateN[seq[tt]]
        if name not in Model_type[iModel[Model_num]][1]:
            print name,v
            print "CHosen",Model_num
            print "Allowed" , Model_type[iModel[Model_num]][1]
            
            raise 
        namesl.append(name)
        
        
        if name in ["Ra0","Ra1","Ra2","Ra3","Ra4","Ra5"]:
            traj[tt][0] =np.random.normal(D[name][0],D[name][1] * scf * R_anisentropy[name][0]) 
            traj[tt][1] =np.random.normal(D[name][0],D[name][1] * scf * R_anisentropy[name][1])
            if ndim == 3:
                traj[tt][2] =np.random.normal(D[name][0],D[name][1] * scf * R_anisentropy[name][2])

            
        if name in ["sRa0","sRa1","sRa2"]:
            #print  sD[name][1] 
            #print start[name]
            traj[tt][0] = sD[name][1] *sds[name][start[name]][0] * scf * sR_anisentropy[name][0]
            traj[tt][1] = sD[name][1] *sds[name][start[name]][1] * scf * sR_anisentropy[name][1]
            if ndim == 3:
                traj[tt][2] = sD[name][1] *sds[name][start[name]][2] * scf  * sR_anisentropy[name][2]
            
            start[name] += 1

        if name in ["Le0","Ri0","Ri1"]:
            
            #theta = dalpha1*(1-2*random.random())
            #Dist = max(0.001,np.random.normal(D[name][0],D[name][1]))
            if ndim == 2:
                x,y = Mus[name][StartMu[name]]
            elif ndim == 3:
                x,y,z = Mus[name][StartMu[name]]
            traj[tt][0] = x
            traj[tt][1] = y      
            if ndim == 3:
                traj[tt][2] = y      

            
            StartMu[name] += 1

            #tot0 += Dist
  
           


    if check_delta:
        print sD
        #print sD
        
        
    def down_grade(seq,Model_num,clean=None):
        
        seq = np.array(seq)

        if clean:
            filteseq = np.ones_like(seq,dtype=np.bool)

            for icat in StateN.keys():
                if np.sum(seq == StateN[icat]) <= clean:
                    filteseq[seq == StateN[icat]] = False
            seq0 = copy.deepcopy(seq)
        
            seq = seq0[filteseq]
            

        realname = list(set(seq.tolist()))

        Nrealcat = [iStateN[ireal] for ireal in realname] 

        realname.sort()
        realname = [iStateN[ir] for ir in realname]
        translate = {StateN[iname]:i for i,iname in enumerate(realname)}
        Nrealcat = realname
        bNrealcat = copy.deepcopy(Nrealcat)
      
                
        
        
        if "Ra1" in Nrealcat and not "Ra0" in Nrealcat:
            seq[seq==StateN["Ra1"]] = StateN["Ra0"]
            
        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname] 
        
        if "Ra2" in Nrealcat and not "Ra1" in Nrealcat:
            if "Ra0" in Nrealcat:
                seq[seq==StateN["Ra2"]] = StateN["Ra1"]
                
            else:
                seq[seq==StateN["Ra2"]] = StateN["Ra0"]
                
                
        if "sRa1" in Nrealcat and not "sRa0" in Nrealcat:
            seq[seq==StateN["sRa1"]] = StateN["sRa0"]
            
        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname] 
        
        if "sRa2" in Nrealcat and not "sRa1" in Nrealcat:
            if "sRa0" in Nrealcat:
                seq[seq==StateN["sRa2"]] = StateN["sRa1"]
                
            else:
                seq[seq==StateN["sRa2"]] = StateN["sRa0"]
        
        if "Ri0" in Nrealcat and not "Le0" in Nrealcat:
            seq[seq==StateN["Ri0"]] = StateN["Le0"]
            
        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname] 
        if "Ri1" in Nrealcat and not "Ri0" in Nrealcat:
            if "Le0" in Nrealcat:
                seq[seq==StateN["Ri1"]] = StateN["Ri0"]
                
            else:
                seq[seq==StateN["Ri1"]] = StateN["Le0"]
                
        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname] 
        Nrealcat.sort()
        #Classing by frequencies
        
        if "Ri0" in Nrealcat and "Le0" in Nrealcat:
            if np.sum(seq==StateN["Ri0"]) >  np.sum(seq==StateN["Le0"]) :
                seq[seq==StateN["Ri0"]] = 1000
                seq[seq==StateN["Le0"]] = StateN["Ri0"]
                seq[seq==1000] = StateN["Le0"]
                
        if "Ri0" in Nrealcat and "Le0" in Nrealcat and "Ri1" in Nrealcat:
            freq = [[np.sum(seq==StateN["Le0"]),"Le0"],
                    [np.sum(seq==StateN["Ri0"]),"Ri0" ],
                    [np.sum(seq==StateN["Ri1"]),"Ri1" ]]
            
            freq.sort()
            freq = freq[::-1]
            seq1 = copy.deepcopy(seq)
            
            seq1[seq == StateN[freq[0][1]]] = StateN["Le0"]
            seq1[seq == StateN[freq[1][1]]] = StateN["Ri0"]
            seq1[seq == StateN[freq[2][1]]] = StateN["Ri1"]
      
            seq = seq1
           
                
                
        #Dowgrading models:

     
        found=False
        for k,v in Model_type.items():
            cats = v[1]
            cats.sort()
            if cats == Nrealcat:
                Model_num = v[0]
                found=True
                break
        if not found:
            print Model_type
            print "CHosen",Model_num
            print bNrealcat
            print Nrealcat
            raise "nimportquei"
        
        if clean:
            seq0[filteseq] = seq
            invfilteseq = np.array([not ifilte for ifilte in filteseq],dtype=np.bool)
            seq0[invfilteseq] = 9
            
            seq = seq0

        return seq,Model_num
    
    
    seq,Model_num = down_grade(seq,Model_num,clean)
    
    
    
    #print translate
    startc = [[0,0]]
    if ndim == 3:
        startc = [[0,0,0]]
    traj = np.cumsum(np.concatenate((startc,traj)),axis=0)
    
    
    #Random nan in seq
    
    nzeros = np.random.randint(0,10)
    Z  = [ i for i,v in enumerate(seq) if v == 9]
    if zeros:
        
        for i in range(nzeros):
            Z.append(np.random.randint(len(seq)-1))
            seq[Z[-1]] = 9
            #traj[Z[-1]:Z[-1]+1,0] = 0  
            #traj[Z[-1]:Z[-1]+1,1] = 0  

        Z= list(set(Z)) 

    
    """
    normed= [copy.deepcopy( np.sqrt(np.sum((traj[1:]-traj[:-1])**2,axis=1)))]
    #print normed[0].shape
    normed.append((traj[1:,0]-traj[:-1,0])/normed[0])
    normed.append((traj[1:,1]-traj[:-1,1])/normed[0])
    
    normed = np.array(normed).T
    
    #print normed.shape
    normed[::,0] = normed[::,0]-np.mean(normed[::,0])
    normed[::,0] /= np.std(normed[::,0])
    """
    ModelN = len(set(namesl))
    return ModelN,Model_num,seq,seq,traj,[],Z

if __name__ == "__main__":
    from prePostTools import get_parameters
    ModelN,Model_num,s,sc,traj,normed,alpha2= generate_traj(200,fight=False,sub=True,
                                                             zeros=False,clean=2,ndim=2,anisentropy=0,
                                                            force_model=3,Mu0 =[[3,0],[1,1],[0]])
    print ModelN,Model_num
    f = figure(figsize=(15,10))
  
    ax = f.add_subplot(141)

    plot(s)
    ax = f.add_subplot(142)
    plot(sc)
    ax = f.add_subplot(143)
    #print traj.shape
    plot_label(traj[::,:2],sc,remove6=9)
    
    print get_parameters(traj,s,1,1,2)

    axis("equal")

    print traj.shape

    print np.sum(s == 3) , np.sum(s == 4) , np.sum(s == 5)

    print alpha2
    print sc
    
  
#print normed.shape


# In[345]:

def fractional_1D(size,H):
    #numpy.random.seed(0)
    N = size
    HH = 2*H
    covariance = numpy.zeros((N,N))
    I = numpy.indices((N,N))

    covariance = abs(I[0]-I[1])
    covariance = (abs(covariance - 1)**HH + (covariance + 1)**HH - 2*covariance**HH)/2.
    
    
    w,v = numpy.linalg.eig(covariance)
    ws = w  **0.25
    v = ws[newaxis,::] * v
    A = np.inner(v,v)
    x = numpy.random.randn((N))
    eta = numpy.dot(A,x)
    xfBm = np.cumsum(eta)
    return np.concatenate([[0],xfBm])

def fractional_1D_slow(size,H):
    #numpy.random.seed(0)
    N = size
    HH = 2*H
    numpy.random.seed(0)
    covariance = numpy.zeros((N,N))
    A = numpy.zeros((N,N))
    for i in range(N):
        for j in range(N):
            d = abs(i-j)
            covariance[i,j] = (abs(d - 1)**HH + (d + 1)**HH - 2*d**HH)/2.
    w,v = numpy.linalg.eig(covariance)
    for i in range(N):
        for j in range(N):
            A[i,j] = sum(math.sqrt(w[k])*v[i,k]*v[j,k] for k in range(N))
    x = numpy.random.randn((N))
    eta = numpy.dot(A,x)
    return np.array([sum(eta[0:i]) for i in range(len(eta)+1)])
    
if __name__ == "__main__":
    print np.sum(fractional_1D(200,0.25)-fractional_1D_slow(200,0.25))
    #%timeit fractional_1D(200,1)
    #%timeit fractional_1D_slow(200,0.5)


# In[22]:

def in_sphere(X,R):
    #print X <= R
    if np.all(X <= R) and np.sum(X**2) <= R**2:
        return True
    return False
def reflect(X,dX):
    norm = np.sum(X**2)**0.5
    norm = X/norm
    parallel = np.sum(norm * dX) * norm
    #print parallel
    return dX-2*parallel


# In[288]:

import scipy
def diffusive(scale,ndim,time,epsilon=1e-7):
        
    mus1 = np.zeros((ndim))
    rho1 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas1 = scale*np.random.random(1) + epsilon


    cov = sigmas1 * np.eye(ndim) + epsilon
    
    return np.cumsum(numpy.random.multivariate_normal(mus1,cov,time),axis=0)


def directed(scale,ndim,time,epsilon=1e-7,delta_sigma_directed=6):
        
    mus1 = scale*(1-2*np.random.random(ndim))
    rho1 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas1 = scale*np.random.random(ndim) + epsilon

    d1 = np.sqrt(np.sum(mus1**2))+epsilon
    if d1 < 0.01:
        mus1 = 0.01*mus1/d1
        
    for indim in range(ndim):
        sigmas1[indim]  = min(d1/delta_sigma_directed,sigmas1[indim])
        
    if ndim == 2:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf],
                           [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf]]
        
    if ndim == 3:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[0]*sigmasm[2]*rhom[1] * scf],
                    [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf,  sigmasm[1]*sigmasm[2]*rhom[2] * scf],
                    [sigmasm[0]*sigmasm[2]*rhom[1] * scf, sigmasm[1]*sigmasm[2]*rhom[2] * scf, sigmasm[2] * scf]]
        
        
    cov = get_covariance(sigmas1,rho1)
    return np.cumsum(numpy.random.multivariate_normal(mus1,cov,time),axis=0)

def accelerated(scale,ndim,time,epsilon=1e-7,delta_sigma_directed=6):
        
    mus1 = scale*(1-2*np.random.random(ndim))
    rho1 = np.random.random(1+(ndim-2)*2) + epsilon
    sigmas1 = scale*np.random.random(ndim) + epsilon
    
    accelerated = 2+2*np.random.rand()

    d1 = np.sqrt(np.sum(mus1**2))+epsilon
    if d1 < 0.01:
        mus1 = 0.01*mus1/d1
        
    for indim in range(ndim):
        sigmas1[indim]  = min(d1/delta_sigma_directed,sigmas1[indim])
        
    if ndim == 2:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf],
                           [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf]]
        
    if ndim == 3:
        scf = 2**0.5
        def get_covariance(sigmasm,rhom):
            return [[sigmasm[0] * scf, sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[0]*sigmasm[2]*rhom[1] * scf],
                    [sigmasm[0]*sigmasm[1]*rhom[0] * scf, sigmasm[1] * scf,  sigmasm[1]*sigmasm[2]*rhom[2] * scf],
                    [sigmasm[0]*sigmasm[2]*rhom[1] * scf, sigmasm[1]*sigmasm[2]*rhom[2] * scf, sigmasm[2] * scf]]
        
        
    cov = get_covariance(sigmas1,rho1)
      
    acc = numpy.random.multivariate_normal(mus1,cov,time) +             mus1 /accelerated * np.array([np.arange(time),np.arange(time),np.arange(time)]).T[::,:ndim]
        
    return np.cumsum(acc,axis=0)

def sinusoidal(scale,ndim,time,epsilon=1e-7):
        
    period = np.random.rand()
    sample_rate = np.random.randint(5,10)
    
    t = np.arange(time)
    traj = np.array([scale * t, scale* np.sin(2*3.14*period*t/sample_rate),np.zeros_like(t)]).T
      
    alpha = 2*3.14 * np.random.rand()
    
    if ndim == 2:
        traj = traj[::,:2]
                
        
    noise = diffusive(scale/4.,ndim,time+1,epsilon=1e-7)
    return random_rot(traj,alpha,ndim) + noise[:-1]-noise[1:]
    

def heart(scale,ndim,time):
    
    
    percent = 1.05 + 0.5*np.random.rand()
    
    real_heart = int(time*percent)
    
    ratio = 0.5*np.random.rand()  #xy ratio
    x = scipy.linspace(-2,2,real_heart/2)
    y1 = scipy.sqrt(1-(abs(x)-1)**2)
    y2 = -3*scipy.sqrt(1-(abs(x[::-1])/2)**0.5)
    
    Y = np.concatenate([y1,y2])
    X  = ratio*np.concatenate([x,x[::-1]])
    
    shift = np.random.randint(0,real_heart)
    Y = np.roll(Y,shift)[:time]
    X = np.roll(X,shift)[:time]
    traj = np.array([X,Y,np.zeros_like(Y)]).T
    
    alpha = 2*3.14 * np.random.rand()
    
    if ndim == 2:
        traj = traj[::,:2]
                    
   
    
    noise = diffusive(scale/800.,ndim,time+1,epsilon=1e-7)
    return scale*random_rot(traj,alpha,ndim) + noise[:-1]-noise[1:]


def subdiffusive(scale,ndim,time):
   alpha = 0.5
   traj = create_random_alpha(time+1,alpha=alpha,ndim=ndim)

   alpha = 2*3.14 * np.random.rand()
   return scale*random_rot(traj,alpha,ndim)[:time]


def fractionnal_brownian(scale,ndim,time):
    alpha = 0.25
    traj = np.array([fractional_1D(time,alpha) for d in range(ndim)]).T

    alpha = 2*3.14 * np.random.rand()
    return scale*random_rot(traj,alpha,ndim)[:time]

def brownian_confined_on_plane(scale,ndim,time):
    
    traj = diffusive(scale,2,time)
    traj = np.concatenate((traj,np.zeros((traj.shape[0],1))),axis=1)

    return random_rot(traj,ndim=3)[:time,:ndim]


def sub_confined_on_plane(scale,ndim,time):
    
    traj = subdiffusive(scale,2,time)
    traj = np.concatenate((traj,np.zeros((traj.shape[0],1))),axis=1)

    return random_rot(traj,ndim=3)[:time,:ndim]
    
def sub_confined_on_plane_0p7(scale,ndim,time,p=0.7):
    
    #p = 0.5 + 0.2 * np.random.rand()

    
    traj = subdiffusive(scale,3,time+1)

    to_cut = range(len(traj) - 1) 
    N = int(len(to_cut) * p )
    np.random.shuffle(to_cut)

    dtraj = traj[1:]-traj[:-1]
    dtraj[to_cut[:N],2]= 0
    
    traj = np.cumsum(dtraj,axis=0)
    return random_rot(traj,ndim=3)[:time,:ndim]

def brownian_confined_on_plane_0p7(scale,ndim,time,p=0.7):
    
    #p = 0.5 + 0.2 * np.random.rand()
    
    traj = diffusive(scale,3,time+1)

    to_cut = range(len(traj) - 1) 
    N = int(len(to_cut) * p )
    np.random.shuffle(to_cut)

    dtraj = traj[1:]-traj[:-1]
    dtraj[to_cut[:N],2]= 0
    
    traj = np.cumsum(dtraj,axis=0)
    return random_rot(traj,ndim=3)[:time,:ndim]

def min_contact(ncontact):
    def dec1(func):
        #print "Min contact",ncontact
        def wrapper(*args,**kwargs):
            kwargs["contact"] = True
            response = [0,0]
            while response[1] < max(1,ncontact):
                response = func(*args,**kwargs)
            
            return response[0]
        return wrapper
    return dec1
    
@min_contact(10)
def brownian_confined_in_sphere(scale,ndim,time,show=False,contact=True):
    
    traj = diffusive(scale,ndim,time+1)
    delta = traj[1:]-traj[:-1]
    
    R = (0.2+0.6*np.random.rand() ) * scale* np.sqrt(time)
    #New 0.3 + 0.1
    #before 0.2 + 0.6
    #print R
    ftraj = [np.zeros((traj.shape[1]))]
    n_contact = 0
    for v in delta:
        #print v
        if in_sphere(ftraj[-1] + v,R):
            ftraj.append(ftraj[-1]+v)
        else:
            n_contact += 1
            r = reflect(ftraj[-1],v)
            #print r,v
            ftraj.append(ftraj[-1]+r)
    if show:
        print n_contact
    traj = np.array(ftraj)
    if contact:
        return random_rot(np.array(traj),ndim=ndim)[:time,:ndim],n_contact
    else:
        return random_rot(np.array(traj),ndim=ndim)[:time,:ndim]



@min_contact(10)
def sub_confined_in_sphere(scale,ndim,time,show=False,contact=True):
    
    traj = subdiffusive(scale,ndim,time+1)
    delta = traj[1:]-traj[:-1]
    
    
    R = (0.9+0.4*np.random.rand() ) * scale* (1.0*time)**0.33
    R = (1.4+1.0*np.random.rand() ) * scale* (1.0*time)**(0.15+0.5/(time-20))

    #print R
    ftraj = [np.zeros((traj.shape[1]))]
    n_contact = 0
    for v in delta:
        #print v
        if in_sphere(ftraj[-1] + v,R):
            ftraj.append(ftraj[-1]+v)
        else:
            n_contact += 1
            r = reflect(ftraj[-1],v)
            #print r,v
            ftraj.append(ftraj[-1]+r)
    
    if show:
        print "N_contact", n_contact
    traj = np.array(ftraj)
    if contact :
        return random_rot(np.array(traj),ndim=ndim)[:time,:ndim], n_contact
    else:
        return random_rot(np.array(traj),ndim=ndim)[:time,:ndim]



    
def generate_traj_general():
    pass

if __name__ == "__main__":
   
    print sinusoidal(1,2,10).shape
    
    #X = sinusoidal(1,3,200)
    #plot(X[::,0],X[::,1])
    
    X = sub_confined_in_sphere(1,3,400,show=True)
   
    #X = brownian_confined_on_plane_0p7(1,3,40)
    #for i in range(10):
    #     brownian_confined_in_sphere(1,3,40,show=True)
    
    #X = fb_confined_in_sphere(1,2,400)

    print X.shape
    
    plot(X[::,0],X[::,2],"-o")


# In[18]:

if __name__ == "__main__":
    ndim=2
    X = sub_confined_on_plane(1,2,100)
    
    #X = fb_confined_in_sphere(1,2,400)

    print X.shape
    
    plot(X[::,0],X[::,1],"-o")

    
    figure()
    
    noise = diffusive(0.4,ndim,len(X)+1,epsilon=1e-7)
    
    dn =  noise[:-1]-noise[1:]
    plot(X[::,0]+dn[::,0],X[::,1]+dn[::,1])


# In[635]:

if __name__ == "__main__":
    print X
    #dot(axis.T,newtraj).T
    #axis = M([1,0,0],np.pi /2)


# In[18]:

#down_grade
#%timeit( generate_traj(200,fight=False,sub=True,force_model=20,clean=4) , 10)
#print 0.37/0.3

if __name__ == "__main__":
    #print X.std()
    print np.sqrt(0.5)
    for i in range(10):
        #5 or 18
        ModelN,Model_num,s,sc,traj,normed,alpha2  = generate_traj(400,fight=False,sub=True,
                                                                  force_model=18,clean=4,
                                                                  check_delta=True,diff_sigma=1.1,Ra0=[0,1.]) 
        #print traj.shape
        #traj = np.concatenate((traj,traj[::,0:1]),axis=1)
        #print traj.shape
        print [[ig[0],ig[2][2]**0.5] for ig in get_parameters(traj,sc,1,1)]


# In[343]:

def MSD(Coords,skip=10,stop=1000,nonOverlap=True):
    #print "MSD doing the calc"
    end = len(Coords[0])
    deb=0
    stop=stop
    indep=2000
    skip = skip
    #nonOverlap=True

    msd = np.zeros(end-deb-stop)
    for k in range(1,end-deb-stop,skip):
        n = 0
        if nonOverlap:
            indep = 0 + k
        else:
            indep = 1
        for i in range(0,end-k-deb,indep):
            start = Coords[::,i]
            pos = Coords[::,i+k]
#            import pdb
#            pdb.set_trace()
            msd[k] += np.sum((start-pos)**2)
            n += 1

        msd[k]  =  msd[k]  /n
    return msd[1::skip]


# In[738]:

if __name__ == "__main__":
    lm = [diffusive,directed,accelerated,heart,sinusoidal,fractionnal_brownian]
    lm += [brownian_confined_in_sphere,brownian_confined_on_plane,fb_confined_in_sphere,fb_confined_on_plane]
    lm += [subdiffusive]
    M2 = ["diff","direct","accel","heart","sinus","frac","b_sphep","B_plane","fb_sphere","fb_plane","sub"]

    for f,n in zip(lm,M2):
        print n
        get_ipython().magic(u'timeit f(1,3,400)')


# In[599]:

if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D

    coord = fb_confined_on_plane(1,2,400)
    #coord = fractionnal_brownian(1,2,400)
    #coord = diffusive(1,3,400)
    
        
    print np.std(coord[1:]-coord[:-1],axis=0)
    msd = MSD(coord[::,:].T,stop=1,skip=1)
    print msd.shape
    
    plot(msd[:50])
    plot(np.arange(50),0.3*np.arange(50))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #scatter(coord[::,0],coord[::,1],coord[::,2])
    #axis("equal")


# In[627]:

if __name__ == "__main__":
    def timeE():
        ModelN,Model_num,s,sc,real_traj,norm,Z = generate_traj(100,sub=True,clean=4,diff_sigma=1.1,ndim=3)


        alpharot = 2*3.14*np.random.random()

        real_traj  = random_rot(real_traj,alpharot,ndim=3)

        alligned_traj,normed,alpha,_ = traj_to_dist(real_traj,ndim=3)
        
    def timeM():
        StateN = {"Ra0": 0,"Ra1":1,"Ra2":2,"Le0":3,"Ri0":4,"Ri1":5,"sRa0":6,"sRa1":7,"sRa2":8}#,"Ra3": 3,"Ra4":4,"Ra5":5,"Le0":6}
        ListState = [0,1,3,5]
        model = one_particle_n_states(ListState0=ListState,StateN=StateN,selfprob=0.4)
        sequence  = model.sample(100)
        for i in sequence:
            pass

        #model.sequence

    get_ipython().magic(u'time timeE()')
    #%prun??
    #%prun -s cumulative timeM()
    #%timeit np.random.multivariate_normal([0,0],[[1,0.1],[1,0]],10)

