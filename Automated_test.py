
# coding: utf-8

# In[1]:

import numpy as np
import pylab as plt
from Tools import generate_traj,random_rot,traj_to_dist

def Brownian_V_separation(graph,range_mu=None,range_len=None,maxlen=800,ndim=2,plot=True,noise_level=0):
    
    if range_mu is None:
        range_mu = np.arange(1,3.1,0.2)
    if range_len is None:
        range_len = range(25,400,25)
        
    #rangemu=[3]
    res = np.zeros((len(range_mu),3+12))
    for Mu,mu in enumerate(range_mu):
        print Mu,mu
        Nt = 100




        for Lenght,l in enumerate(range_len):
            if l %2 == 1:
                    l = l- 1

            Traj = []
            Real_traj = []
            S = []
            for n in range(Nt):
                size = l
                Ra0 = [0,1.]
                succeed = False
                g = 0
                while not succeed or g > 10:
                    try:
                        #print "gen"
                        g += 1
                        ModelN,Model_num,s,sc,real_traj,norm,alpha2 = generate_traj(size,lower_selfprob=0.9,
                                                                                    fight=False,diff_sigma=2,
                                                                                    deltav=0.1,zeros=False,
                                                                                    delta_sigma_directed=0.1,
                                                                                    force_model = 3,
                                                                                    anisentropy=0,
                                                                                    Ra0=Ra0,Mu0=[[mu,0],[1,1],[0]],
                                                                                    ndim=ndim)
                        if Model_num == 3:
                            break


                    except IndexError:
                        print "Failed"
                        succeed = False

                #R = get_parameters(real_traj,s,1,1,2)
                #print R[0][2][1] ,  R[1][2] [0]
                alpharot = 2*3.14*np.random.random()

                real_traj2  = random_rot(real_traj,alpharot,ndim=ndim)
                if noise_level != 0:
                    real_traj += np.random.normal(0,noise_level,real_traj.shape)
                alligned_traj,normed,alpha,_ = traj_to_dist(real_traj2[::,:ndim],ndim=ndim)

                Traj.append(normed)

                Real_traj.append(real_traj2)
                S.append(s)



            #print np.array(Traj).shape,np.array(Traj)[::,:l,::].shape
            res1 = graph.predict({"input1":np.array(Traj)[::,:l,::]})
            cat = res1["category"]

            res[Mu,Lenght] = np.sum(np.argmax(cat,-1) == [3]) / 1.0 / Nt

    if plot:
        #,cmap=plt.get_cmap("cool")
        plt.imshow(res[::-1,::],interpolation="None",extent=(0,400,1,3),aspect=200)
        plt.colorbar()
        plt.savefig("separation-unfilterer.png")
        
    return res


# In[19]:

import numpy as np
import pylab as plt
from Tools import generate_traj,random_rot,traj_to_dist

def Brownian_Brownian_separation(graph,range_b=None,range_len=None,maxlen=800,ndim=2,plot=True):
    
    if range_b is None:
        range_b = np.arange(2,10,1)
    if range_len is None:
        range_len = range(25,400,25)
        
    #rangemu=[3]
    res = np.zeros((len(range_b),3+12))
    for Mu,mu in enumerate(range_b):
        print Mu,mu
        Nt = 100




        for Lenght,l in enumerate(range_len):
            if l %2 == 1:
                    l = l- 1

            Traj = []
            Real_traj = []
            S = []
            for n in range(Nt):
                size = l
                Ra0 = [0,1.]
                succeed = False
                g = 0
                while not succeed or g > 10:
                    try:

                        g += 1
                        ModelN,Model_num,s,sc,real_traj,norm,alpha2 = generate_traj(size,lower_selfprob=0.9,
                                                                                    fight=False,diff_sigma=2,
                                                                                    deltav=0.1,zeros=False,
                                                                                    delta_sigma_directed=0.1,
                                                                                    force_model = 2,
                                                                                    anisentropy=0,
                                                                                    Ra0=Ra0,Mu0=[[mu,0],[1,1],[0]],
                                                                                    Ra1 = [0,mu],
                                                                                    ndim=ndim)
                        if Model_num == 2:
                            break


                    except IndexError:
                        print "Failed"
                        succeed = False

                #R = get_parameters(real_traj,s,1,1,2)
                #print R[0][2][1] ,  R[1][2] [0]
                alpharot = 2*3.14*np.random.random()

                real_traj2  = random_rot(real_traj,alpharot,ndim=ndim)

                alligned_traj,normed,alpha,_ = traj_to_dist(real_traj2[::,:ndim],ndim=ndim)

                Traj.append(normed)

                Real_traj.append(real_traj2)
                S.append(s)



            #print np.array(Traj).shape,np.array(Traj)[::,:l,::].shape
            res1 = graph.predict({"input1":np.array(Traj)[::,:l,::]})
            cat = res1["category"]

            res[Mu,Lenght] = np.sum(np.argmax(cat,-1) == [2]) / 1.0 / Nt

    if plot:
        #,cmap=plt.get_cmap("cool")
        plt.imshow(res[::-1,::],interpolation="None",extent=(0,400,range_b[0],range_b[-1]),aspect=50)
        plt.colorbar()
        plt.savefig("separation-unfilterer-B.png")
        
    return res

if __name__ == "__main__":
    pass
    #Brownian_Brownian_separation("test")


# In[20]:

import numpy as np
import pylab as plt
from Tools import generate_traj,random_rot,traj_to_dist

def V_V_separation(graph,range_mu=None,range_len=None,maxlen=800,ndim=2,plot=True):
    
    if range_mu is None:
        range_mu = np.arange(1.5,3.1,0.2)
    if range_len is None:
        range_len = range(25,400,25)
        
    #rangemu=[3]
    res = np.zeros((len(range_mu),3+12))
    for Mu,mu in enumerate(range_mu):
        print Mu,mu
        Nt = 100




        for Lenght,l in enumerate(range_len):
            if l %2 == 1:
                    l = l- 1

            Traj = []
            Real_traj = []
            S = []
            for n in range(Nt):
                size = l
                Ra0 = [0,1.]
                succeed = False
                g = 0
                while not succeed or g > 10:
                    try:
                        #print "gen"
                        g += 1
                        ModelN,Model_num,s,sc,real_traj,norm,alpha2 = generate_traj(size,lower_selfprob=0.9,
                                                                                    fight=False,diff_sigma=2,
                                                                                    deltav=0.1,zeros=False,
                                                                                    delta_sigma_directed=0.1,
                                                                                    force_model = 4,
                                                                                    anisentropy=0,
                                                                                    Mu1=[[1,0],[1,1],[0]],
                                                                                    Mu0=[[mu,0],[1,1],[0]],
                                                                                    ndim=ndim)
                        if Model_num == 4:
                            break


                    except IndexError:
                        print "Failed"
                        succeed = False

                #R = get_parameters(real_traj,s,1,1,2)
                #print R[0][2][1] ,  R[1][2] [0]
                alpharot = 2*3.14*np.random.random()

                real_traj2  = random_rot(real_traj,alpharot,ndim=ndim)

                alligned_traj,normed,alpha,_ = traj_to_dist(real_traj2[::,:ndim],ndim=ndim)

                Traj.append(normed)

                Real_traj.append(real_traj2)
                S.append(s)



            #print np.array(Traj).shape,np.array(Traj)[::,:l,::].shape
            res1 = graph.predict({"input1":np.array(Traj)[::,:l,::]})
            cat = res1["category"]

            res[Mu,Lenght] = np.sum(np.argmax(cat,-1) == [4]) / 1.0 / Nt

    if plot:
        #,cmap=plt.get_cmap("cool")
        plt.imshow(res[::-1,::],interpolation="None",extent=(0,400,1,3),aspect=200)
        plt.colorbar()
        plt.savefig("separation-unfilterer.png")
        
    return res

