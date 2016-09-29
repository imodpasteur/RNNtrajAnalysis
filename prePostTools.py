
# coding: utf-8

# In[1]:

M1 = ["D","Dv","D-D","D-DvL","DvR-DvL","D-D-D", "D-D-DvL","D-DvL-DvR","D-D-DvL-DvR",
          "DvL-DvR-DvR1","D-DvL-DvR-DvR1","D-D-DvL-DvR-DvR1","sD",
                      "D-sD",
                      "sD-sD",
                      "sD-DvL",                     
                      "sD-D-D",
                      "sD-sD-D",
                      "sD-sD-sD",      
                      "sD-D-DvL",
                      "sD-sD-DvL",              
                      "sD-DvL-DvR",
                      "sD-D-DvL-DvR",  
                      "sD-sD-DvL-DvR",                 
                      "sD-DvL-DvR-DvR1",
                      "sD-D-DvL-DvR-DvR1",
                      "sD-sD-DvL-DvR-DvR1"]

M1b = ["D","Dv","2D","D-Dv","2Dv","3D", "2D-Dv","D-2Dv","2D-2Dv",
          "3Dv","D-3Dv","2D-3Dv","sD",
                      "D-sD",
                      "2sD",
                      "sD-Dv",                     
                      "sD-2D",
                      "2sD-D",
                      "3sD",      
                      "sD-D-Dv",
                      "2sD-DvL",              
                      "sD-2Dv",
                      "sD-D-2Dv",  
                      "2sD-2Dv",                 
                      "sD-3Dv",
                      "sD-D-3Dv",
                      "2sD-3Dv"]


M0 = ["D","Dv","D-D","D-DvL","DvR-DvL","D-D-D", "D-D-DvL","D-DvL-DvR","DvL-DvR-DvR1","D-D-DvL-DvR",
          "D-DvL-DvR-DvR1","D-D-DvL-DvR-DvR1"]


# In[11]:

def get_parameters(traj,segmentation,pixel,time,ndim):
    deltas = traj[1:] - traj[:-1]
    
    cat = list(set(segmentation))

    cat.sort()
    
    #print deltas.shape
    Vals = []
    for subcat in cat:
        if np.isnan(subcat):
            continue
        wh = np.array(segmentation) == subcat
        Mean = np.mean(deltas[wh],axis=0)


        #print "inside",subcat,sum(wh)
        if subcat in [0,1,2,6,7,8]:
            Vals.append([subcat,np.sum(wh),[Mean,np.mean(np.sqrt(deltas[wh]**2),axis=0),
                                 np.mean(np.sum(deltas[wh]**2,axis=1))*pixel**2/time/(2*ndim),
                                            np.mean(np.sum(deltas[wh]**2,axis=1))**0.5]])
        else:
            Mean = np.mean(deltas[wh],axis=0)
            Vals.append([subcat,np.sum(wh),[Mean,np.mean(np.sqrt((deltas[wh]-Mean)**2),axis=0),
                                 np.mean(np.sum((deltas[wh]-Mean)**2,axis=1))*pixel**2/time/(2*ndim),
                                            np.mean(np.sum((deltas[wh]-Mean)**2,axis=1))**0.5]])
        
            
    return Vals

if __name__ == "__main__":
    
    traj = [[0,0,0],[1,1,1],[1,1,1],[2,2,2],[2,2,2]]
    traj = np.array(traj)
    segmentation = np.array([0,0,0,np.nan])
    print get_parameters(traj,segmentation,1,1,2)
    


# In[16]:

import numpy as np
from numpy import histogram
import copy


def clean(traj_proba,class_traj,fight=True,options=[],sub=False,append_steady=False):
    StateN = {"Ra0": 0,"Ra1":1,"Ra2":2,"Le0":3,"Ri0":4,"Ri1":5,"sRa0":6,"sRa1":7,"sRa2":8,"Steady":9}#,"Ra3": 3,"Ra4":4,"Ra5":5,"Le0":6}
    iStateN = {v:k for k,v in StateN.items()}
    
    if not sub:
        StateN["Steady"] = 6
    
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
    
    if fight:
        Model_type["DvL-DvR-DvR1"][0] = 8
        Model_type["D-D-DvL-DvR"][0] = 9
        
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

        
    if append_steady:
        for k in Model_type.keys():
            Model_type[k][-1].append("Steady")
    
    cats = np.argmax(traj_proba,axis=-1)
    if 4 in cats and not 3 in cats:
        traj_proba[::,3:5] = traj_proba[::,3:5][::,::-1]
        
    cats = np.argmax(traj_proba,axis=-1)
    
    if 5 in cats and not 3 in cats:
        t2 = traj_proba.copy()
        t2[::,3] = traj_proba[::,5]
        t2[::,5] = traj_proba[::,3]
        traj_proba = t2
        
    elif 5 in cats and not 4 in cats:
        traj_proba[::,4:6] = traj_proba[::,4:6][::,::-1]
        
    iModel = {v[0]:k for k,v in Model_type.items()}
    

    inside =[StateN[name] for name in Model_type[iModel[class_traj]][1]]

    #print inside
    traj_max_restricted = np.argmax(traj_proba[::,inside],axis=-1)
    seq = [inside[itraj] for itraj in traj_max_restricted]
    
    
    
    return np.array(seq)


# In[3]:

def traj_to_dist(traji,bins=20,random_rotation=0,ndim=2):
    

    trajip = traji[1:,:ndim]-traji[:-1,:ndim]
    norm = np.sqrt(np.sum(trajip[::]**2,axis=1))[::,np.newaxis]
    trajip /= norm

      
    avoid = np.isnan(trajip) 
    trajip[avoid] = 0

    theta = np.arccos(trajip[::,0])
    
    theta [ trajip[::,1] <0] = 2*3.14 - theta [ trajip[::,1] <0] 
    
    
    theta = theta[~avoid[::,0]]
    #print trajip[:10,0]
    
    #hist(theta,bins=bins)

    
    count,pos = histogram(theta,bins=bins)
    
    directionp = pos[np.argmax(count)]
    
    """ 
    count2,pos2 = histogram(theta,bins=2*bins)
    
    directionp2 = pos2[np.argmax(count2)]
    """
    count4,pos4 = histogram(theta,bins=4*bins)
    
    directionp4 = pos4[np.argmax(count4)]
    
    
    #print directionp, directionp4
    #if abs(directionp4 - directionp) < 0.3 or  abs(directionp4 - directionp) >6.:
    #    directionp = directionp4
    
    #print directionp, abs(directionp4 - directionp)
    #print directionp
    #rirectionp=3.14*2-0.1
    if ndim == 2:
        axis = [[np.cos(directionp),np.cos(directionp+3.14/2)],
           [np.sin(directionp),np.sin(directionp+3.14/2)]]
    if ndim == 3:
        axis = [[np.cos(directionp),np.cos(directionp+3.14/2),0],
           [np.sin(directionp),np.sin(directionp+3.14/2),0],
               [0,0,1]]
    axis=np.array(axis)
    
    #print axis

    avoid = np.isnan(traji) 

    newtraj =  (traji-np.mean(traji[~avoid[::,0]].T,axis=1)).T 

    alligned_traj = np.dot(axis.T,newtraj).T
    
    dist = np.sqrt( np.sum((alligned_traj[1:]-alligned_traj[:-1])**2,axis=1))
    #print coeff,latent
    
    normed= [copy.deepcopy(dist),copy.deepcopy(dist)]

    normed.append((alligned_traj[1:,0]-alligned_traj[:-1,0])/dist)
    normed.append((alligned_traj[1:,1]-alligned_traj[:-1,1])/dist)
    if ndim == 3:
        normed.append((alligned_traj[1:,2]-alligned_traj[:-1,2])/dist)

    
    normed.append([ len(dist)/100. for i in range(len(dist))])
    
    
    normed = np.array(normed).T
    
    
    normed[::,0] = normed[::,0]-np.mean(normed[::,0])
    normed[::,0] /= np.std(normed[::,0])
    normed[::,1] /=  np.std(normed[::,1])
    
    #Zero = normalized - mean
    #One = normalized
    
    normed[np.isnan(normed)] = 0
    
    return alligned_traj,normed,directionp,axis


# In[2]:

from Toolv1 import traj_to_dist2
def filter_same(traj):
    new_traj = []
    
    #i = 1
    zeros=[]
    falsezeros = []
    nnan = []
    nn = 0
    addzero=False
    #First replace single point by nan
    
    for i in range(1,len(traj)-1):
        
        if np.isnan(traj[i-1][0]) and np.isnan(traj[i+1][0]):
            traj[i] = traj[i] * np.nan
            
    #Then remove the nan from the trajectory
    for i in range(len(traj)):
       
        if not np.isnan(traj[i][0]):
            if addzero:
                addzero = False
                falsezeros.append(True)
                zeros.append(len(new_traj))
                
                nnan.append(nn)

            new_traj.append(traj[i])
            falsezeros.append(False)

            nn = 0
            
                
        else:
            addzero = True
            nn += 1
    if addzero:
        addzero = False
        falsezeros.append(True)
        nnan.append(nn)
        
        #Not needed
        zeros.append(len(new_traj))

    return np.array(new_traj),zeros,nnan,falsezeros


def clean_initial_trajectory(traj0,v=1):
    
    traj,zeros,nans,falsezeros = filter_same(traj0)

    added0 = False
    #We need trajectory that can be subsampled by two
    if len(traj) % 2 == 0:
        #traj = np.concatenate((traj,np.zeros_like(traj[0:1,::])),axis=0)
        traj = np.concatenate((traj,traj[-1:,::]),axis=0)
        added0 = True
        zeros.append(len(traj)-1)

    if v == 1:
        alligned_traj,normed,directionp,_ = traj_to_dist(traj)
    if v == 2:
        alligned_traj,normed = traj_to_dist2(traj)
        
    for i in zeros:
        if i == 0:
            continue
        if i-1 < len(normed):
            normed[i-1,::] = 0 
    
    return traj,alligned_traj,normed,zeros,nans,added0
            
def put_back_nan(cat,zeros,nans):

    
    for izeros in zeros:
        if izeros <= len(cat) and izeros != 0:
            cat[izeros-1] = np.nan

    if 0 in zeros:
        cat.insert(0,np.nan)
        
    
    if len(zeros)>=1 and len(cat) <= zeros[-1]:
        cat.append(np.nan)

        
    #Middle nan should be increased by one
    for i,iz in enumerate(zeros):
        if iz != 0 and iz  < len(cat)-1:
            #print iz,len(cat) 
            nans[i] += 1
    #print nans , zeros
    start = 0
    newcat = []
    for c in cat:
        if np.isnan(c):
            try:
                for inan in range(nans[start]):
                    newcat.append(np.nan)
            except:
                newcat.append(np.nan)
            start += 1
        else:
            newcat.append(c)
    return newcat

def test_nans():
    trajs =  [ [[0.2,0.2,0.2],[False,False]],
               [[0.2,0.2,0.2,0.2],[False,False,False]],
               [[0.2,0.2,np.nan,0.2],[False,True,True]],
               [[0.2,0.2,0.2,0.2],[False,False,False]],
                [[np.nan,np.nan,0.2,1,0.3,np.nan,0.4,0.4,0.2,np.nan],
                [True,True,False,False,True,True,False,False,True]],
               [[np.nan,np.nan,0.2,1,0.3,np.nan,0.4,0.4,0.2,np.nan,np.nan],
                [True,True,False,False,True,True,False,False,True,True]],
              [[np.nan,np.nan,0.2,1,0.3,np.nan,0.4,0.4,0.2,0.2,np.nan],
                [True,True,False,False,True,True,False,False,False,True]],
              [[0.2,1,0.3,np.nan,0.4,0.4,0.2,0.2,np.nan],
                [False,False,True,True,False,False,False,True]],
               [[0.2,1,0.3,np.nan,np.nan,0.4,0.4,0.2,0.2,np.nan],
                [False,False,True,True,True,False,False,False,True]],
               [[0.2,1,0.3,np.nan,np.nan,0.4,0.4,0.2,0.2,0.2,np.nan],
                [False,False,True,True,True,False,False,False,False,True]],
                   [[0.2,1,0.3,0.1,np.nan,np.nan,np.nan,0.4,0.4,0.2,0.2,0.2,np.nan],
                [False,False,False,True,True,True,True,False,False,False,False,True]],
                [[0.2,1,0.3,0.1,np.nan,np.nan,np.nan,0.4,0.4,0.2,0.2,0.2,np.nan,0.1],
                [False,False,False,True,True,True,True,False,False,False,False,True,True]],
               [[0.2,1,0.3,0.1,np.nan,np.nan,np.nan,0.4,0.4,0.2,0.2,0.2,np.nan,0.1,np.nan],
                [False,False,False,True,True,True,True,False,False,False,False,True,True,True]],
              [[0.2,1,0.3,0.1,np.nan,0.1,np.nan,0.4,np.nan,0.2,0.2,0.2,np.nan,0.1,np.nan],
                [False,False,False,True,True,True,True,True,True,False,False,True,True,True]]]
              
              
    for traj0,isnan in trajs:
        traj1 = np.array([traj0,traj0]).T
        
        traj,alligned_traj,normed,zeros,nans,added0 = clean_initial_trajectory(traj1)

        #graph produce a cat with same size than normed
        cat = [1 for i in range(len(normed))]
        
        if added0:
            zeros.pop(-1)
            cat.pop(-1)

            
        #print "Before"
        #print cat
        cat = put_back_nan(cat,zeros,nans)
        
        
        #print traj0
        #print np.isnan(cat),isnan
        assert np.all(np.isnan(cat) == np.array(isnan))
        
if __name__ == "__main__":
    test_nans()

