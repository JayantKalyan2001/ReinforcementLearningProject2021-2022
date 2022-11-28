from os import O_APPEND
from typing import Type
from dataset import *
from train import *
from evaluate import *
from explain import *
import random 
import warnings
import imblearn
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore",category=DeprecationWarning)
ramObs=onp.asarray(load_ram_obs_from_dataset(r"datasets/smallset"))
Obs=onp.asarray(load_atari_obs_from_dataset(r"datasets/smallset"))
qvalueDS=onp.asarray(load_q_values_from_dataset(r"datasets/smallset"))
print(type(Obs))
#print(onp.shape(ramObs))
#for f in ramObs:
 #   print("ram obs ",f)
print(imblearn.__version__)
actionDs=load_discrete_actions_from_dataset(r"datasets/smallset")
#print("actions shape",onp.shape(actionDs))
#trainingOutput=train_action_dt(ramObs,actionDs,max_depth=5)
#trainingEnsembleOutput=train_ensemble_action_dt(ramObs,actionDs,max_depth=5)
#trainingMLPOutput=trainMLPActionClassifier(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLP=trainDeepMLPActionClassifier(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLPOverSampled=trainDeepMLPOverSampledActionClassifier(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLPQvalue=trainDeepMLPQValueRegressor(ram_obs_dataset=ramObs,qval_dataset=qvalueDS)
#deepMLPMarkov=trainDeepMarkovMLPActionClassifier(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLPMarkovOverSampled=trainDeepMarkovMLPActionClassifierOverSampled(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLPMarkovOverSampledNoPrevAction=trainDeepMarkovMLPActionClassifierOverSampledNoPreviousAction(ram_obs_dataset=ramObs,actions_dataset=actionDs)
deepMLPMarkovNoPrevAction=trainDeepMarkovMLPActionClassifierNoPreviousAction(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#deepMLPMarkovUnderSampledNoPrevAction=trainDeepMarkovMLPActionClassifierUnderSampledNoPreviousAction(ram_obs_dataset=ramObs,actions_dataset=actionDs)
#qvalueOutput=train_q_values_dt(ramObs,qvalueDS,max_depth=10)
print("ramobs type",type(ramObs))
print ("qvalue ds shape",onp.shape(qvalueDS))
#ensembleTreeStruct=trainingEnsembleOutput[1]
#treeStruct=trainingOutput[0]
#mlp=trainingMLPOutput

#print("dt results",results)
#print("mlp results",mlpResults)
def DTPolicyfn(ramObs):
    tree= treeStruct
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=tree.predict(ramObs)
    treeType="ConventionalDT"
    return action,treeType
def EnsemblePolicyfn(ramObs):
    tree= ensembleTreeStruct
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=tree.predict(ramObs)
    treeType="EnsembleDT"
    return action,treeType
def RandomPolicyfn(ramObs):
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=random.randint(1,3)
    treeType="Random"
    return action,treeType

def mlpPolicyfn(ramObs):
    NeuralNet= mlp
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=onp.asarray(NeuralNet.predict(ramObs))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="MLP"
    return action,treeType


def deepmlpPolicyfn(ramObs):
    NeuralNet= deepMLP
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=onp.asarray(NeuralNet.predict(ramObs))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLP"
    return action,treeType

def deepmlpOversampledPolicyfn(ramObs):
    NeuralNet= deepMLPOverSampled
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=onp.asarray(NeuralNet.predict(ramObs))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPOverSampled"
    return action,treeType    

def deepMarkovMLPPolicyfn(ramObs,prev,prevAction):
    NeuralNet= deepMLPMarkov
    #print("inputted RAM obs shape ",onp.shape(ramObs)
    #print("prev:",prev,'\n')
    #print("ramNow:",ramObs,'\n')
    prevAction=OneHotEncoder(prevAction,4)
   # print("previous action",prevAction,'\n')
    mergedRAM=onp.concatenate([prev,ramObs,prevAction],axis=1)
    #print("shape merged",onp.shape(mergedRAM))
    action=onp.asarray(NeuralNet.predict(mergedRAM))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPMarkov"
    return action,treeType

def deepMarkovMLPOverSampledPolicyfn(ramObs,prev,prevAction):
    NeuralNet= deepMLPMarkovOverSampled
    #print("inputted RAM obs shape ",onp.shape(ramObs)
    #print("prev:",prev,'\n')
    #print("ramNow:",ramObs,'\n')
    prevAction=OneHotEncoder(prevAction,4)
   # print("previous action",prevAction,'\n')
    mergedRAM=onp.concatenate([prev,ramObs,prevAction],axis=1)
    #print("shape merged",onp.shape(mergedRAM))
    action=onp.asarray(NeuralNet.predict(mergedRAM))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPMarkovOverSampled"
    return action,treeType
def deepMarkovMLPOverSampledNoPrevActionPolicyfn(ramObs,prev):
    NeuralNet= deepMLPMarkovOverSampledNoPrevAction
    #print("inputted RAM obs shape ",onp.shape(ramObs)
    #print("prev:",prev,'\n')
    #print("ramNow:",ramObs,'\n')
    #prevAction=OneHotEncoder(prevAction,4)
   # print("previous action",prevAction,'\n')
    mergedRAM=onp.concatenate([prev,ramObs],axis=1)
    #print("shape merged",onp.shape(mergedRAM))
    action=onp.asarray(NeuralNet.predict(mergedRAM))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPMarkovOverSampledNoPrevAction"
    return action,treeType
def deepMarkovMLPUnderSampledNoPrevActionPolicyfn(ramObs,prev):
    NeuralNet= deepMLPMarkovUnderSampledNoPrevAction
    #print("inputted RAM obs shape ",onp.shape(ramObs)
    #print("prev:",prev,'\n')
    #print("ramNow:",ramObs,'\n')
    #prevAction=OneHotEncoder(prevAction,4)
   # print("previous action",prevAction,'\n')
    mergedRAM=onp.concatenate([prev,ramObs],axis=1)
    #print("shape merged",onp.shape(mergedRAM))
    action=onp.asarray(NeuralNet.predict(mergedRAM))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPMarkovUnderSampledNoPrevAction"
    return action,treeType
def deepMarkovMLPNoPrevActionPolicyfn(ramObs,prev):
    NeuralNet= deepMLPMarkovNoPrevAction
    #print("inputted RAM obs shape ",onp.shape(ramObs)
    #print("prev:",prev,'\n')
    #print("ramNow:",ramObs,'\n')
    #prevAction=OneHotEncoder(prevAction,4)
   # print("previous action",prevAction,'\n')
    mergedRAM=onp.concatenate([prev,ramObs],axis=1)
    #print("shape merged",onp.shape(mergedRAM))
    action=onp.asarray(NeuralNet.predict(mergedRAM))
    #print("mlp action",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPMarkovNoPrevAction"
    return action,treeType
def deepmlpQvaluePolicy(ramObs):
    NeuralNet= deepMLPQvalue
    #print("inputted RAM obs shape ",onp.shape(ramObs))
    action=onp.asarray(NeuralNet.predict(ramObs))
    #print("mlp qvalue:",action)
    #print("action type policy fn",type(onp.where(action[0]==onp.amax(action[0]))))
    action=onp.asarray(onp.where(action[0]==onp.amax(action[0]))).item()
    #print("mlp action",action)
    treeType="DeepMLPQvalueRegressor"
    return action,treeType

deepMLPMarkovScores=[]
deepMLPMarkovScoresNoPrevActions=[]
randomScores=[]
DeepMarkovMLPOverSampledScores=[]
DeepMarkovMLPOverSampledScoresNoPrevAction=[]
DTScores=[]
EnsembleScores=[]

JustRAMObsMLP=[]
JustRAMObsDeepMLP=[]
JustRAMObsDeepMLPOversampled=[]
JustRAMObsDeepMLPQValueRegressor=[]
DeepMarkovMLPUnderSampledScoresNoPrevAction=[]
for i in range(1):
    outputPerformanceDeepMarkovMLPNoPrev=markov_decision_tree_performance_no_prev_action('Breakout',deepMarkovMLPNoPrevActionPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    deepMLPMarkovScoresNoPrevActions.append(outputPerformanceDeepMarkovMLPNoPrev[0])
    """"
    outputPerformanceDT=decision_tree_performance('Breakout',DTPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Decision Tree done")
    outputPerformanceEnsemble=decision_tree_performance('Breakout',EnsemblePolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Ensemble Tree done")
    outputPerformanceRandom=decision_tree_performance('Breakout',RandomPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random done")
    outputPerformanceMLP=decision_tree_performance('Breakout',mlpPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs MLP (3 layer) done")
    outputPerformanceDeepMLP=decision_tree_performance('Breakout',deepmlpPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs Deep (5 layer) MLP done")
    outputPerformanceDeepMLPOversampled=decision_tree_performance('Breakout',deepmlpOversampledPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs Deep (5 layer) MLP done")
    outputPerformanceDeepMLPQvalReg=decision_tree_performance('Breakout',deepmlpQvaluePolicy,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs Deep (5 layer) Q-value regression MLP done")
    outputPerformanceDeepMarkovMLP=markov_decision_tree_performance('Breakout',deepMarkovMLPPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs + Previous State + Previous Action Deep MLP done")
    outputPerformanceDeepMarkovMLPOverSampled=markov_decision_tree_performance('Breakout',deepMarkovMLPOverSampledPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs + Previous State + Previous Action Deep Oversampled MLP done")
    outputPerformanceDeepMarkovMLPOverSampledNoPrevAction=markov_decision_tree_performance_no_prev_action('Breakout',deepMarkovMLPOverSampledNoPrevActionPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    print("Random Obs + Previous State + No Previous Action Deep Oversampled MLP done")
    #outputPerformanceDeepMarkovMLPUnderSampledNoPrevAction=markov_decision_tree_performance_no_prev_action('Breakout',deepMarkovMLPUnderSampledNoPrevActionPolicyfn,[25,78,2,3,1,34,109,405,30,21,22,32,33,57,58])
    #print("Random Obs + Previous State + No Previous Action Deep Undersampled MLP done")
    #print("DT scores",outputPerformanceDT[0])
    #print("Ensemble scores",outputPerformanceEnsemble[0])

    deepMLPMarkovScores.append(outputPerformanceDeepMarkovMLP[0])
    randomScores.append(outputPerformanceRandom[0])
    DeepMarkovMLPOverSampledScores.append(outputPerformanceDeepMarkovMLPOverSampled[0])
    DeepMarkovMLPOverSampledScoresNoPrevAction.append(outputPerformanceDeepMarkovMLPOverSampledNoPrevAction[0])

    DTScores.append(outputPerformanceDT[0])
    EnsembleScores.append(outputPerformanceEnsemble[0])
    JustRAMObsMLP.append(outputPerformanceMLP[0])
    JustRAMObsDeepMLP.append(outputPerformanceDeepMLP[0])
    JustRAMObsDeepMLPOversampled.append(outputPerformanceDeepMLPOversampled[0])
    JustRAMObsDeepMLPQValueRegressor.append(outputPerformanceDeepMLPQvalReg[0])
    #DeepMarkovMLPUnderSampledScoresNoPrevAction.append(outputPerformanceDeepMarkovMLPUnderSampledNoPrevAction[0])
    #print("MLP scores",outputPerformanceMLP[0])
    #print("Deep MLP scores",outputPerformanceDeepMLP[0])
    #print("Deep MLP Q Value regressor scorees",outputPerformanceDeepMLPQvalReg[0])
"""

plt.hist(actionDs)
plt.show() 
""""
print("Deep Markov MLP:\n")
for d in deepMLPMarkovScores:
    print(d,'\n')

print("Deep Markov MLP OverSampled:\n")
for d in DeepMarkovMLPOverSampledScores:
    print(d,'\n')

print("Deep Markov MLP OverSampled No Prev Action:\n")
for d in DeepMarkovMLPOverSampledScoresNoPrevAction:
    print(d,'\n')

print("Deep Markov MLP UnderSampled No Prev Action:\n")
for d in DeepMarkovMLPUnderSampledScoresNoPrevAction:
    print(d,'\n')
print("Decision tree :\n")
for d in DTScores:
    print(d,'\n')
print("Ensemble tree:\n")
for d in EnsembleScores:
    print(d,'\n')
print("Normal MLP:\n")
for d in JustRAMObsMLP:
    print(d,'\n')
print("Deep MLP:\n")
for d in JustRAMObsDeepMLP:
    print(d,'\n')
print("Deep MLP Oversampled:\n")
for d in JustRAMObsDeepMLPOversampled:
    print(d,'\n')
print("Deep MLP Q-Value Regressor:\n")
for d in JustRAMObsDeepMLPQValueRegressor:
    print(d,'\n')

print("Random:\n")
for d in randomScores:
    print(d,'\n')
"""
print("Deep Markov MLP No Prev Actions:\n")
for d in deepMLPMarkovScoresNoPrevActions:
    print(d,'\n')
#print("dqn obs shape", onp.shape(Obs),"ram",onp.shape(ramObs))
#print("dt obs shape",onp.shape(outputPerformanceDT[2]),"ram",onp.shape(ramObs))
#print("dqn obs",type(Obs))
#print("dt obs",type(outputPerformanceDT[2]))
#dtObs=onp.asarray(outputPerformanceDT[2])
#ensembleObs=onp.asarray(outputPerformanceEnsemble[2])
#randObs=onp.asarray(outputPerformanceRandom[2])
#print("dt obs onp asaaray",type(dtObs))
#animate_observations(obs=Obs,savePath=r"videos/dqn.mp4")
#animate_observations(obs=dtObs,savePath=r"videos/decisiontree.mp4")
#animate_observations(obs=ensembleObs,savePath=r"videos/ensemble.mp4")
#animate_observations(obs=ensembleObs,savePath=r"videos/random.mp4")
#for e in explanation:
 #   print(e)
#localSaveFunc=lambda  datasetSize,maxDepth,agentName,distPercentage : "/mnt/c/Users/jk5g19/Documents/explainable-atari-ram/{}/{}/{}/{}".format(datasetSize,maxDepth,agentName,distPercentage)
#localSaveFunc=lambda  datasetSize,maxDepth,agentName,distPercentage,envName: f"datasets/dataset-{datasetSize}/{agentName}-{envName}"
#evaluate_hyperparameters(save_folder_fn=localSaveFunc,network_root_folder=network_root_folderJK)