
# coding: utf-8

# In[ ]:

"""
Mac install:
step 1 install docker toolbox:
https://docs.docker.com/mac/step_one/

step 2 launch Docker Quickstart Terminal

step 3: to test docker installation type in the Docker terminal:
docker run hello-world

step 4: test of the installation of rnn software:
    step 4_1:
    create a folder to store the result on your computer ex:

    ex /mypersonalfiles/result
    
    step 4_2:
    run on Docker terminal: 
    docker run  -v /mypersonalfiles/result:/src/results -e="THEANO_FLAGS='device=cpu'" jeammimi/docker-keras-rnn  python main.py 2 Tracking_results-inspected__MinFrames_30.mat


    (where you replace  /mypersonalfiles/result by the folder that you created on step 4_1):
    
    (it may take a while to download the docker image (1 Gb))
    
    step 4_3:
    
    check your folder /mypersonalfiles/result there shourd be some result files
    

step 5 running it on your on experiment:

    docker run  -v /mypersonalfiles/mytraj.mat:/src/test.mat -v /mypersonalfiles/result:/src/results -e="THEANO_FLAGS='device=cpu'" jeammimi/docker-keras-rnn  python main.py 2 test.mat

    (where you replace  /mypersonalfiles/result by the folder that you created on your computer ,
    and where you replace  /mypersonalfiles/mytraj.mat by the files which contain your trajectories)
    
    
SPECIFIC issues:
    mac  the file and the directory must be in /Users and not in /Volumes otherwise
         there would be an error: test.mat is a directory.
         error: test.mat is a directory can also happend if there is a mistake in the path of the file.
"""


# In[1]:

"""
sudo docker daemon
docker run -ti --dns=157.99.64.65 --name running jeammimi/docker-keras-rnn:v5 /bin/sh
docker cp src running:/src           #to copy file from here to container
docker commit -m "My modif" running jeammimi/docker-keras-rnn:v2   #to save running container

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

docker run  -v /home/jarbona/docker-app/src/Tracking_results-inspected__MinFrames_30.mat:/src/test.mat -v /home/jarbona/restult_tmp/:/src/results -e="THEANO_FLAGS='device=cpu'" docker-keras4  python main.py 2 test.mat

run   -v /home/jarbona/docker-app/test/:/src/results -e="THEANO_FLAGS='device=cpu'" jeammimi/docker-keras-rnn:v9  python main.py 2 results --batch 1

docker push jeammimi/docker-keras-rnn:v2
"""


import warnings
warnings.filterwarnings("ignore")

import argparse

#theano.config.device="cpu"

parser = argparse.ArgumentParser()
parser.add_argument("ndim", help="number of dimension",
                    type=int)
parser.add_argument("trackfile", help="trackfile or directory (if batch mode)")

parser.add_argument("--res", help="result folder",default="results/")

parser.add_argument("--sub", help="include subdiffusif model",default="1")

parser.add_argument("--batch", help="include subdiffusif model",default="0")

#parser.add_argument("--format", help="possible format mat or json ",default="mat")


args = parser.parse_args()

ndim = args.ndim
filetoopen = args.trackfile
res_folder = args.res
sub = args.sub
batch = args.batch

if sub == "1":
    sub = True
else:
    sub = False

if batch == "1":
    batch = True
else:
    batch = False

aformat = "mat"

print ""
print "RNN with following parameters:"
if not sub:
    print " - No",
else:
    print " -",
print "Subdiffusive motion " ,

if not sub:
    print " (to add diffusive motion add the option --sub 1)"
else:
    print " (to remove diffusive motion add the option --sub 0)"

print " - %i dimensions"%ndim
print ""
print "Loading model"
    


# In[2]:

#loading network
from scipy.io import loadmat,savemat
import numpy as np
import theano
theano.config.mode="FAST_COMPILE"

from Specialist_layer import return_three_bis


graph9 = return_three_bis(ndim=ndim)
print "Running model"


# In[3]:

from scipy.io import loadmat,savemat
from prePostTools import get_parameters,M1,M0
import copy


def save_on_mat(name,step_categorie,categories,traj,px,fr,ndim):
    
    #fich = "template.mat"
    #Mp = loadmat(fich,squeeze_me=False)
    Mp= {'results':{}}
    Mp['results']["PrM"] = categories
    Mp['results']['track'] = traj.T
    Mp['results']['steps'] = np.array(traj[1:,::]-traj[:-1,::]).T
    
    remove_nan = np.array(copy.deepcopy(step_categorie),dtype=np.float)
    
    remove_nan[ remove_nan == 9] =  remove_nan[ remove_nan == 9] * np.nan
    
    Mp['results']["ML_states"] = remove_nan

    
    res = get_parameters(traj,remove_nan,pixel=px,time=1./fr,ndim=ndim)
    mu_emit = []
    mu_real = []
    sigma_emit = []
    D_emit = []
    mu_emit = []
    for scat in res:
        if np.isnan(scat[0]):
            continue
        
        #print "\nScat" ,scat
        
        mu_real.append(scat[2][0])
        if scat[0] in [0,1,2,6,7,8]:
            mu_emit.append(np.zeros_like(scat[2][0]))
        else:
            mu_emit.append(scat[2][0])
        sigma_emit.append(scat[2][-1])
        D_emit.append(scat[2][-2])

    
    Mp['results']['ML_params'] = {}
    Mp['results']['ML_params']["mu_emit"] = np.array(mu_emit).T
    Mp['results']['ML_params']["mu_real"] = np.array(mu_real).T
    Mp['results']['ML_params']["sigma_emit"] = np.array(sigma_emit).T
    Mp['results']['ML_params']["D_emit"] = np.array(D_emit).T


    if name is not None:
        savemat(name,Mp)
    
    return Mp
    


# In[22]:

from prePostTools import clean_initial_trajectory,put_back_nan,clean,traj_to_dist
from keras.utils import generic_utils
import json
import glob
import os


"""
ndim = 2
filetoopen ="/home/jarbona/Downloads/v3_crop-1_MinFr30.mat"
res_folder = "./"
sub = True
batch = False
aformat = "mat"
"""

if sub:
    graph9.load_weights("saved_weights/three_bilayer_sub_bis")

else:
    graph9.load_weights("saved_weights/three_layer_trained_on_no_sub_anisentropy0_diffsigma1_delta_sigma_directed3")



    
def process_one_file(filetoopen,localfolder,filetowrite):
    trajs = []
    M = loadmat(filetoopen)


    pred_RNNs = []
    pred_RNN_cats = []

    add = { "tool": "RNN",
             "mu_emit_unit":"pixel",
             "mu_real_unit":"pixel",
            "sigma_emit_unit":"pixel",
            "D_emit_unit":"Assuming your unit for the pixel size was nanometer, in mu^2/s"}
    if not M.has_key("analysisInfo"):
        M["analysisInfo"] = {}
    else:
        tmp = {}
        for k in M["analysisInfo"].dtype.names:
            tmp[k] = M["analysisInfo"][k][0][0][0][0] 
        M["analysisInfo"] = tmp
    for k,v in add.items():
            M["analysisInfo"][k] = v

    if sub:
         M["analysisInfo"]["PrM_labels"] = np.array(M1,dtype=np.object)
    else:
         M["analysisInfo"]["PrM_labels"] = np.array(M0,dtype=np.object)

    px = M["analysisInfo"]["pixelSize"]/ 1000. #In micro meter
    fr = M["analysisInfo"]["frameRate"]



    M["results"] = []
    #NewField = ["ML_states","PrM","steps",'ML_params']

    #Copy old fields
    #M["tracksProc"] #= {field:M["tracksProc"][field][0] for field in M["tracksProc"].dtype.names}

    #print M["tracksProc"]["pos"].shape
    #Add new fields
    #tmp = {}
    #for field in NewField:
    #    M["tracksProc"][field] = np.empty(len(M["tracksProc"]["pos"]), dtype=np.object)

    ntraj = len(M["tracksProc"][0]["pos"])
    progbar = generic_utils.Progbar(ntraj)


    for itraj,traj0 in enumerate(M["tracksProc"][0]["pos"]):

        #print traj
        #traj0 = traj

        #print traj0.shape
        traj,alligned_traj,normed,zeros,nans,added0 = clean_initial_trajectory(traj0)
        trajs.append(traj)
        #print normed.shape
        #pred0 = graph9.predict({"input1":normed[newaxis,::,::]})


        pred0 = graph9.predict({"input1":normed[np.newaxis,::,::]},  batch_size=1)

        pred_RNN = pred0["output"]
        pred_RNN_cat = pred0["category"]

        #Inverse these 
        if not sub:
            #print pred_RNN_cat.shape
            pred_RNN_cat[0,8:10] = pred_RNN_cat[0,8:10][::-1]
            pred_RNN_cat = pred_RNN_cat[0,:len(M0)]


        pred_RNN = pred_RNN[0]
        pred_RNN_cat = pred_RNN_cat

        if added0:
            zeros.pop(-1)
            traj = traj[:-1,::]
            pred_RNN = pred_RNN[:-1]

        pred_RNNs.append(pred_RNN)
        pred_RNN_cats.append(pred_RNN_cat)



        cat =  pred_RNN
        fight = False
        if not sub:
            fight = True
        cat = clean(cat,np.argmax(pred_RNN_cat),fight=fight,sub=sub,append_steady=False)

        cat = cat.tolist()

        cat = put_back_nan(cat,zeros,nans)

        assert(len(cat) == len(traj0)-1)


        #save_on_mat(res_folder +"res%i.mat"%(itraj+1),cat,pred_RNN_cat,traj0)
        Mp = save_on_mat(None,cat,pred_RNN_cat,traj0,px=px,fr=fr,ndim=ndim)



        M["results"].append( Mp["results"])

        progbar.add(1)

        #print cat

    #M["tracksProc"] = np.array(M["tracksProc"],dtype=)
    
    final = os.path.join(localfolder,filetowrite)

    if aformat == "mat":
        final += ".mat"
        filetowrite += ".mat"
        savemat(final,M)
    else:
        final += ".json"
        filetowrite += ".json"
        with open(final,"w") as f:
            f.write(json.dumps(M))
        #print cat
        #print cat

    print "Result writen in ",filetowrite


# In[21]:

if not batch:
    print "processing ",filetoopen
    process_one_file(filetoopen,res_folder , "RNN_track_analysis")

else:

    liste_file = glob.glob(os.path.join(filetoopen,"") + "*.mat")
    liste_file.sort()
    
    if len(liste_file) == 0:
        print "No file found in ", filetoopen
        print "check the folder specified in -v option"
        raise
    if not os.path.exists(os.path.join(filetoopen,"RNN")):
        print "The files " , liste_file 
        print "where found, but" 
        print "You have to create a RNN folder in the directory where your trajectories are stored"
        raise
    for filet in liste_file:
        
        print "processing ",filet
        final_file = os.path.join("RNN",os.path.split(filet)[1])
        process_one_file(filet,filetoopen,final_file[:-4]+"_RNN")
        print 


# In[ ]:



