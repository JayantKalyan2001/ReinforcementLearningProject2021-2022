"""
Imitation learning for decision trees to learn either the action or q-value of a pre-train atari
"""
import functools
from heapq import merge
from multiprocessing.dummy import active_children
import pickle
from tabnanny import verbose
from typing import Sequence, Tuple, List, Dict, Any, Callable
import tensorflow as tf
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import supersuit
from  explain import *
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier 
import gc
import imblearn
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.metrics import precision_recall_fscore_support as score
from collections import Counter
#from sklearn.preprocessing import OneHotEncoder

from explainable_atari_ram.dataset import create_directory

evaluation_seeds = [514750, 415793, 700628, 680697, 233137, 27270, 428496, 322621,
                    495605, 108341, 441261, 610813, 71067, 81225, 990322, 64677,
                    706392, 607638, 123158, 80431]  # np.random.randint(0, 1_000_000, 20)
def getPrevRAMOBs(ramObs,singularRAMObs):
    ramObs=onp.asarray(ramObs)
    singularRAMObs=onp.asarray(singularRAMObs)
    if onp.argmax(onp.all(ramObs==singularRAMObs,axis=1))==0:
        return onp.zeros(128)
    else:
        return ramObs[onp.argmax(onp.all(ramObs==singularRAMObs,axis=1))-1]

def getPreviousAction(actionsDS,action):
    if onp.argmax(onp.all(actionsDS==action,axis=0))==0:
        return 0
    else:
        return actionsDS[onp.argmax(onp.all(actionsDS==action,axis=0))-1]

def GetActionFromQValue(qvalDS):
    optAction=[]
    for q in qvalDS:
        opt=[]
        for val in q:
            if val == max(q):
                opt.append(1)
            else:
                opt.append(0)
        #print(opt)
        optAction.append(opt)
    
    return onp.asarray(optAction)



def OneHotEncoder(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = onp.array(data).reshape(-1)
    return onp.eye(nb_classes)[targets]

def action_dt_policy(ram_obs: onp.ndarray, decision_tree: DecisionTreeClassifier) -> int:
    """
    A policy function for an action based decision tree

    :param ram_obs: the input ram observations for the decision tree
    :param decision_tree: The decision tree for making the decision
    :return: The policy action
    """
    return int(decision_tree.predict(ram_obs)[0])

def cross_entropy_accuracy(Prediction2d,Actual2d):
    accuracy=0
    for prediction1d,actual1d in Prediction2d,Actual2d:
        accuracy+=accuracy_score(prediction1d,actual1d)
    return accuracy/len(Prediction2d)

def q_value_dt_policy(ram_obs: onp.ndarray, decision_trees: List[DecisionTreeRegressor]) -> int:
    """
    A policy function for a q-values based decision tree

    :param ram_obs: the input ram observations for the decision tree
    :param decision_trees: List of decision tree for making the decision
    :return: The policy action
    """
    return int(onp.argmax([decision_tree.predict(ram_obs) for decision_tree in decision_trees]))


def decision_tree_performance(env_name: str, policy_fn: Callable[[onp.ndarray], int], seeds: Sequence[int],
                              epsilon: float = 0.01) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Evaluates the performance of an approximate network performance that takes the RAM observations of the agent

    :param env_name: The environment name
    :param policy_fn: Policy function taking the RAM observation and return an action
    :param seeds: The evaluation seeds
    :param epsilon: The probability that the action taken is random
    :return: List of total rewards taken over seeds
    """
    env = supersuit.frame_stack_v1(gym.wrappers.AtariPreprocessing(gym.make(f'{env_name}NoFrameskip-v4')))
    total_rewards, total_steps = onp.zeros(len(seeds)), onp.zeros(len(seeds))
   # env.reset()
   # ob, reward, done, _ = env.step(0)
    obs=[]
    for pos, seed in enumerate(seeds):
        env.seed(seed)
        _, done, rng = env.reset(), False, jax.random.PRNGKey(seed=seed)
        ob, reward, done, _ = env.step(0)  
        while not done:
            rng, epsilon_rng, action_rng = jax.random.split(rng, num=3)
            #print("get RAM from ale shape ",onp.shape(env.unwrapped.ale.getRAM().reshape(1, -1)))
            action = jnp.where(jax.random.uniform(epsilon_rng) <= epsilon,
                               env.action_space.sample(),
                               policy_fn(env.unwrapped.ale.getRAM().reshape(1, -1))[0])
            typeTree=policy_fn(env.unwrapped.ale.getRAM().reshape(1, -1))[1]
           # print("treeType:",typeTree)
           # print("action type",type(action))
            action=onp.asscalar(action)
            #print("action",action)
            ob, reward, done, _ = env.step(action)
            env.env.ale.saveScreenPNG(f'Screenshots/{typeTree}/Frame{seed}_{total_steps[pos]}.png')
            #print("ob shape dt: ", onp.shape(ob))
            obs.append(ob)
            total_rewards[pos] += reward
            total_steps[pos] += 1
    #print("total steps",total_steps)
    return total_rewards,total_steps,obs
def markov_decision_tree_performance(env_name: str, policy_fn: Callable[[onp.ndarray], int], seeds: Sequence[int],
                              epsilon: float = 0) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Evaluates the performance of an approximate network performance that takes the RAM observations of the agent

    :param env_name: The environment name
    :param policy_fn: Policy function taking the RAM observation and return an action
    :param seeds: The evaluation seeds
    :param epsilon: The probability that the action taken is random
    :return: List of total rewards taken over seeds
    """
    env = supersuit.frame_stack_v1(gym.wrappers.AtariPreprocessing(gym.make(f'{env_name}NoFrameskip-v4')))
    total_rewards, total_steps = onp.zeros(len(seeds)), onp.zeros(len(seeds))
   # env.reset()
   # ob, reward, done, _ = env.step(0)
    obs=[]
    for pos, seed in enumerate(seeds):
        env.seed(seed)
        _, done, rng = env.reset(), False, jax.random.PRNGKey(seed=seed)
        ob, reward, done, _ = env.step(0)
        currentRAM=onp.zeros(128).reshape(1,-1)
        previousRAM=onp.zeros(128).reshape(1,-1)
        previousAction=0
        #print("get prev RAM",onp.shape(previousRAM))  
        while not done:
            rng, epsilon_rng, action_rng = jax.random.split(rng, num=3)
            previousRAM=currentRAM
           # print("updated prev")
            currentRAM=env.unwrapped.ale.getRAM().reshape(1, -1)
           # print("updated current")
            policyOutput=policy_fn(currentRAM,previousRAM,previousAction)
            action = jnp.where(jax.random.uniform(epsilon_rng) <= epsilon,
                               env.action_space.sample(),
                               policyOutput[0])
            typeTree=policyOutput[1]
           # print("treeType:",typeTree)
           # print("action type",type(action))
            action=onp.asscalar(action)
            #print("action",action)
            ob, reward, done, _ = env.step(action)
            previousAction=action
            env.env.ale.saveScreenPNG(f'Screenshots/{typeTree}/Frame{seed}_{total_steps[pos]}.png')
            #print("ob shape dt: ", onp.shape(ob))
            obs.append(ob)
            total_rewards[pos] += reward
            total_steps[pos] += 1
    #print("total steps",total_steps)
    return total_rewards,total_steps,obs

def markov_decision_tree_performance_no_prev_action(env_name: str, policy_fn: Callable[[onp.ndarray], int], seeds: Sequence[int],
                              epsilon: float = 0) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Evaluates the performance of an approximate network performance that takes the RAM observations of the agent

    :param env_name: The environment name
    :param policy_fn: Policy function taking the RAM observation and return an action
    :param seeds: The evaluation seeds
    :param epsilon: The probability that the action taken is random
    :return: List of total rewards taken over seeds
    """
    env = supersuit.frame_stack_v1(gym.wrappers.AtariPreprocessing(gym.make(f'{env_name}NoFrameskip-v4')))
    total_rewards, total_steps = onp.zeros(len(seeds)), onp.zeros(len(seeds))
   # env.reset()
   # ob, reward, done, _ = env.step(0)
    obs=[]
    for pos, seed in enumerate(seeds):
        env.seed(seed)
        _, done, rng = env.reset(), False, jax.random.PRNGKey(seed=seed)
        ob, reward, done, _ = env.step(0)
        currentRAM=onp.zeros(128).reshape(1,-1)
        previousRAM=onp.zeros(128).reshape(1,-1)
        #previousAction=0
       # print("get prev RAM",onp.shape(previousRAM))  
        while not done:
            rng, epsilon_rng, action_rng = jax.random.split(rng, num=3)
            previousRAM=currentRAM
           # print("updated prev")
            currentRAM=env.unwrapped.ale.getRAM().reshape(1, -1)
           # print("updated current")
            policyOutput=policy_fn(currentRAM,previousRAM)
            action = jnp.where(jax.random.uniform(epsilon_rng) <= epsilon,
                               env.action_space.sample(),
                               policyOutput[0])
            typeTree=policyOutput[1]
           # print("treeType:",typeTree)
           # print("action type",type(action))
            action=onp.asscalar(action)
            #print("action",action)
            ob, reward, done, _ = env.step(action)
           # previousAction=action
            env.env.ale.saveScreenPNG(f'Screenshots/{typeTree}/Frame{seed}_{total_steps[pos]}.png')
            #print("ob shape dt: ", onp.shape(ob))
            obs.append(ob)
            total_rewards[pos] += reward
            total_steps[pos] += 1
    #print("total steps",total_steps)
    return total_rewards,total_steps,obs

def trainMLPActionClassifier(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    training_ram_obs, validation_ram_obs, training_actions, validation_actions = \
        train_test_split(ram_obs_dataset, actions_dataset, train_size=0.75, test_size=0.25)
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_ram_obs[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_ram_obs))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_ram_obs,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_ram_obs)))
    action_mlp.evaluate(validation_ram_obs,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_ram_obs),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp
def trainDeepMLPQValueRegressor(ram_obs_dataset: onp.ndarray, qval_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:

    optimalActionArray=GetActionFromQValue(qval_dataset)
    #print("optimal shape",onp.shape(optimalActionArray))
    training_ram_obs, validation_ram_obs, training_optimal, validation_optimal= \
        train_test_split(ram_obs_dataset,optimalActionArray , train_size=0.75, test_size=0.25)
    print("optimal training shape",onp.shape(training_optimal))
    print("optimal validation shape",onp.shape(validation_optimal))
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_ram_obs[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(training_optimal[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    #print("training ram obs shapes",onp.shape(training_ram_obs))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_ram_obs,training_optimal,epochs=100)
    #print("prediction shape",onp.shape(action_mlp.predict(training_ram_obs)))
    action_mlp.evaluate(validation_ram_obs,validation_optimal,verbose=2)
    precision, recall, fscore, support = score(onp.argmax(validation_optimal,axis=1), onp.argmax(action_mlp.predict(validation_ram_obs),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp 

def trainDeepMLPActionClassifier(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    training_ram_obs, validation_ram_obs, training_actions, validation_actions = \
        train_test_split(ram_obs_dataset, actions_dataset, train_size=0.75, test_size=0.25)
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_ram_obs[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_ram_obs))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_ram_obs,trainingActionsEncoded,epochs=100)
    #print("prediction shape",onp.shape(action_mlp.predict(training_ram_obs)))
    action_mlp.evaluate(validation_ram_obs,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_ram_obs),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def trainDeepMLPOverSampledActionClassifier(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    randomOverSampler = RandomOverSampler(sampling_strategy="not majority")
    ram_obs_dataset,actions_dataset=randomOverSampler.fit_resample(ram_obs_dataset,actions_dataset)
    training_ram_obs, validation_ram_obs, training_actions, validation_actions = \
        train_test_split(ram_obs_dataset, actions_dataset, train_size=0.75, test_size=0.25)
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_ram_obs[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_ram_obs))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_ram_obs,trainingActionsEncoded,epochs=100)
    #print("prediction shape",onp.shape(action_mlp.predict(training_ram_obs)))
    action_mlp.evaluate(validation_ram_obs,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_ram_obs),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def trainDeepMarkovMLPActionClassifier(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    prevRamObs=[]
    prevActions=[]
    for r in ram_obs_dataset:
        prevRamObs.append(getPrevRAMOBs(ram_obs_dataset,r))
        gc.collect
    for a in actions_dataset:
        prevActions.append(getPreviousAction(actions_dataset,a))
        gc.collect
    prevRamObs=onp.asarray(prevRamObs)
    prevActions=onp.asarray(prevActions)
    prevActionsEncoded=OneHotEncoder(prevActions,4)
    print("prev RAM Obs",onp.shape(prevRamObs))
    print("prev action",onp.shape(prevActionsEncoded))
   # for i in range(len(ram_obs_dataset)):
   #    print("ram obs",ram_obs_dataset[i],"prev ram obs",prevRamObs[i])
    mergedRamObs=onp.concatenate([prevRamObs,ram_obs_dataset,prevActionsEncoded],axis=1)
    print("merge Obs",onp.shape(mergedRamObs))
    training_merged, validation_merged, training_actions, validation_actions = \
        train_test_split(mergedRamObs, actions_dataset, train_size=0.75, test_size=0.25)

    #print("prev ram obs training",onp.shape(prevRamObsTraining))
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_merged[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_merged))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_merged,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_merged)))
    action_mlp.evaluate(validation_merged,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_merged),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def trainDeepMarkovMLPActionClassifierOverSampledNoPreviousAction(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    This uses KNN to generate similar data points for similar points for the minority classifications
    No Previous action is taken into account due to possible missclassification cascade 
    """
    prevRamObs=[]
    #prevActions=[]
    for r in ram_obs_dataset:
        prevRamObs.append(getPrevRAMOBs(ram_obs_dataset,r))
        gc.collect
    #for a in actions_dataset:
     #   prevActions.append(getPreviousAction(actions_dataset,a))
      #  gc.collect
    prevRamObs=onp.asarray(prevRamObs)
    #prevActions=onp.asarray(prevActions)
    #prevActionsEncoded=OneHotEncoder(prevActions,4)
    print("prev RAM Obs",onp.shape(prevRamObs))
    #print("prev action",onp.shape(prevActionsEncoded))
   # for i in range(len(ram_obs_dataset)):
   #    print("ram obs",ram_obs_dataset[i],"prev ram obs",prevRamObs[i])
    mergedRamObs=onp.concatenate([prevRamObs,ram_obs_dataset],axis=1)
    print("merge Obs before",onp.shape(mergedRamObs))
    print("before oversampling",Counter(actions_dataset))
    randomOverSampler = RandomOverSampler(sampling_strategy="not majority")
    mergedRamObs,actions_dataset=randomOverSampler.fit_resample(mergedRamObs,actions_dataset)
    print("merge Obs after",onp.shape(mergedRamObs))
    print("after oversampling",Counter(actions_dataset))
    training_merged, validation_merged, training_actions, validation_actions = \
        train_test_split(mergedRamObs, actions_dataset, train_size=0.75, test_size=0.25)

    #print("prev ram obs training",onp.shape(prevRamObsTraining))
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_merged[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_merged))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_merged,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_merged)))
    action_mlp.evaluate(validation_merged,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_merged),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def trainDeepMarkovMLPActionClassifierNoPreviousAction(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    This uses KNN to generate similar data points for similar points for the minority classifications
    No Previous action is taken into account due to possible missclassification cascade 
    """
    prevRamObs=[]
    #prevActions=[]
    for r in ram_obs_dataset:
        prevRamObs.append(getPrevRAMOBs(ram_obs_dataset,r))
        gc.collect
    #for a in actions_dataset:
     #   prevActions.append(getPreviousAction(actions_dataset,a))
      #  gc.collect
    prevRamObs=onp.asarray(prevRamObs)
    #prevActions=onp.asarray(prevActions)
    #prevActionsEncoded=OneHotEncoder(prevActions,4)
    print("prev RAM Obs",onp.shape(prevRamObs))
    #print("prev action",onp.shape(prevActionsEncoded))
   # for i in range(len(ram_obs_dataset)):
   #    print("ram obs",ram_obs_dataset[i],"prev ram obs",prevRamObs[i])
    mergedRamObs=onp.concatenate([prevRamObs,ram_obs_dataset],axis=1)
   # print("merge Obs before",onp.shape(mergedRamObs))
   # print("before oversampling",Counter(actions_dataset))
   # randomOverSampler = RandomOverSampler(sampling_strategy="not majority")
  #  mergedRamObs,actions_dataset=randomOverSampler.fit_resample(mergedRamObs,actions_dataset)
  #  print("merge Obs after",onp.shape(mergedRamObs))
  #  print("after oversampling",Counter(actions_dataset))
    training_merged, validation_merged, training_actions, validation_actions = \
        train_test_split(mergedRamObs, actions_dataset, train_size=0.75, test_size=0.25)

    #print("prev ram obs training",onp.shape(prevRamObsTraining))
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_merged[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_merged))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_merged,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_merged)))
    action_mlp.evaluate(validation_merged,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_merged),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def trainDeepMarkovMLPActionClassifierUnderSampledNoPreviousAction(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    This uses KNN to generate similar data points for similar points for the minority classifications
    No Previous action is taken into account due to possible missclassification cascade 
    """
    prevRamObs=[]
    #prevActions=[]
    for r in ram_obs_dataset:
        prevRamObs.append(getPrevRAMOBs(ram_obs_dataset,r))
        gc.collect
    #for a in actions_dataset:
     #   prevActions.append(getPreviousAction(actions_dataset,a))
      #  gc.collect
    prevRamObs=onp.asarray(prevRamObs)
    #prevActions=onp.asarray(prevActions)
    #prevActionsEncoded=OneHotEncoder(prevActions,4)
    print("prev RAM Obs",onp.shape(prevRamObs))
    #print("prev action",onp.shape(prevActionsEncoded))
   # for i in range(len(ram_obs_dataset)):
   #    print("ram obs",ram_obs_dataset[i],"prev ram obs",prevRamObs[i])
    mergedRamObs=onp.concatenate([prevRamObs,ram_obs_dataset],axis=1)
    print("merge Obs before",onp.shape(mergedRamObs))
    print("before oversampling",Counter(actions_dataset))
    randomUnderSampler = RandomUnderSampler(sampling_strategy="majority")
    mergedRamObs,actions_dataset=randomUnderSampler.fit_resample(mergedRamObs,actions_dataset)
    print("merge Obs after",onp.shape(mergedRamObs))
    print("after oversampling",Counter(actions_dataset))
    training_merged, validation_merged, training_actions, validation_actions = \
        train_test_split(mergedRamObs, actions_dataset, train_size=0.75, test_size=0.25)

    #print("prev ram obs training",onp.shape(prevRamObsTraining))
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_merged[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_merged))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_merged,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_merged)))
    action_mlp.evaluate(validation_merged,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_merged),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp
def trainDeepMarkovMLPActionClassifierOverSampled(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray,)-> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    This uses KNN to generate similar data points for similar points for the minority classifications 
    """
    prevRamObs=[]
    prevActions=[]
    for r in ram_obs_dataset:
        prevRamObs.append(getPrevRAMOBs(ram_obs_dataset,r))
        gc.collect
    for a in actions_dataset:
        prevActions.append(getPreviousAction(actions_dataset,a))
        gc.collect
    prevRamObs=onp.asarray(prevRamObs)
    prevActions=onp.asarray(prevActions)
    prevActionsEncoded=OneHotEncoder(prevActions,4)
    print("prev RAM Obs",onp.shape(prevRamObs))
    print("prev action",onp.shape(prevActionsEncoded))
   # for i in range(len(ram_obs_dataset)):
   #    print("ram obs",ram_obs_dataset[i],"prev ram obs",prevRamObs[i])
    mergedRamObs=onp.concatenate([prevRamObs,ram_obs_dataset,prevActionsEncoded],axis=1)
    print("merge Obs before",onp.shape(mergedRamObs))
    print("before oversampling",Counter(actions_dataset))
    randomOverSampler = RandomOverSampler(sampling_strategy="not majority")
    mergedRamObs,actions_dataset=randomOverSampler.fit_resample(mergedRamObs,actions_dataset)
    print("merge Obs after",onp.shape(mergedRamObs))
    print("after oversampling",Counter(actions_dataset))
    training_merged, validation_merged, training_actions, validation_actions = \
        train_test_split(mergedRamObs, actions_dataset, train_size=0.75, test_size=0.25)

    #print("prev ram obs training",onp.shape(prevRamObsTraining))
    trainingActionsEncoded,validationActionsEncoded=OneHotEncoder(training_actions,4),OneHotEncoder(validation_actions,4)
    trainingActionsEncoded=onp.asarray(trainingActionsEncoded)
    validationActionsEncoded=onp.asarray(validationActionsEncoded)
    action_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(260, activation='relu'),
  tf.keras.layers.Dense(4)
])
    mlpPred=action_mlp(training_merged[:1]).numpy()
    tf.nn.softmax(mlpPred).numpy()
    lossFn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lossFn(trainingActionsEncoded[:1],mlpPred)
    action_mlp.compile(optimizer='adam',loss=lossFn,
              metrics=['accuracy'])
    print("training ram obs shapes",onp.shape(training_merged))
    #print("training actions",onp.shape(training_actions))
    #print("ram obs type",type(training_ram_obs))
    #print("actions",type(training_actions))

    action_mlp.fit(training_merged,trainingActionsEncoded,epochs=100)
    print("prediction shape",onp.shape(action_mlp.predict(training_merged)))
    action_mlp.evaluate(validation_merged,validationActionsEncoded,verbose=2)
    precision, recall, fscore, support = score(validation_actions, onp.argmax(action_mlp.predict(validation_merged),axis=1))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return action_mlp

def train_action_dt(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray, max_depth: int = None,
                    env_name: str = None) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Trains an action decision tree policy using sklearn DecisionTreeClassifier

    :param ram_obs_dataset: The RAM observation dataset
    :param actions_dataset: The action dataset
    :param max_depth: The max depth of the decision tree
    :param env_name: The environment name
    :return: The decision tree along with the training, validation and environment results
    """
    training_ram_obs, validation_ram_obs, training_actions, validation_actions = \
        train_test_split(ram_obs_dataset, actions_dataset, train_size=0.75, test_size=0.25)

    action_dt = DecisionTreeClassifier(max_depth=max_depth).fit(training_ram_obs, training_actions)

    results = {
        'training': {
            'accuracy': accuracy_score(action_dt.predict(training_ram_obs), training_actions),
            'mse': action_dt.score(training_ram_obs, training_actions)
        },
        'validation': {
            'accuracy': accuracy_score(action_dt.predict(validation_ram_obs), validation_actions),
            'mse': action_dt.score(validation_ram_obs, validation_actions)
        }
    }

    precision, recall, fscore, support = score(validation_actions, action_dt.predict(validation_ram_obs))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print(results)
    if env_name:
        total_rewards, total_steps = decision_tree_performance(
            env_name, functools.partial(action_dt_policy, decision_tree=action_dt), evaluation_seeds)
        results['environment-rewards'], results['environment-steps'] = list(total_rewards), list(total_steps)

    return action_dt, results


def train_ensemble_action_dt(ram_obs_dataset: onp.ndarray, actions_dataset: onp.ndarray, max_depth: int = None,
                             env_name: str = None) -> Tuple[HistGradientBoostingClassifier, DecisionTreeClassifier,
                                                            Dict[str, Any]]:
    """
    Trains an action decision tree and ensemble decision tree policy using sklearn HistGradientBoostingClassifier

    :param ram_obs_dataset: The RAM observation dataset
    :param actions_dataset: The action dataset
    :param max_depth: The max depth of the decision tree
    :param env_name: The environment name
    :return: The decision tree along with the training, validation and environment results
    """
    training_ram_obs, validation_ram_obs, training_actions, validation_actions = \
        train_test_split(ram_obs_dataset, actions_dataset, train_size=0.75, test_size=0.25)

    # Train the histogram gradient boosted classifier
    ensemble_action_dt: HistGradientBoostingClassifier = HistGradientBoostingClassifier()
    ensemble_action_dt.fit(training_ram_obs, training_actions)

    # Ensemble results
    ensemble_results = {
        'training': {
            'accuracy': accuracy_score(ensemble_action_dt.predict(training_ram_obs), training_actions),
            'score': ensemble_action_dt.score(training_ram_obs, training_actions)
        },
        'validation': {
            'accuracy': accuracy_score(ensemble_action_dt.predict(validation_ram_obs), validation_actions),
            'score': ensemble_action_dt.score(validation_ram_obs, validation_actions)
        }
    }

    # Environment results
    if env_name:
        total_rewards, total_steps = decision_tree_performance(
            env_name, functools.partial(action_dt_policy, decision_tree=ensemble_action_dt), evaluation_seeds)
        ensemble_results['environment-rewards'] = list(total_rewards)
        ensemble_results['environment-steps'] = list(total_steps)

    # Train the decision tree
    action_dt, results = train_action_dt(ram_obs_dataset, ensemble_action_dt.predict(ram_obs_dataset),
                                         max_depth=max_depth, env_name=env_name)
    results['ensemble'] = ensemble_results

    precision, recall, fscore, support = score(validation_actions, ensemble_action_dt.predict(validation_ram_obs))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print(results['ensemble'])
    return ensemble_action_dt, action_dt, results


def train_q_values_dt(ram_obs_dataset: onp.ndarray, q_values_dataset: onp.ndarray, max_depth: int = None,
                      env_name: str = None) -> Tuple[List[DecisionTreeRegressor], Dict[str, Any]]:
    """
    Trains a q-value decision tree policy using sklearn DecisionTreeRegressor

    :param ram_obs_dataset: The RAM observation dataset
    :param q_values_dataset: The q-values dataset
    :param max_depth: The max depth of the decision tree
    :param env_name: The environment name
    :return: A list of decision tree along with the training, validation and environment results
    """
    training_ram_obs, validation_ram_obs, training_q_values, validation_q_values = \
        train_test_split(ram_obs_dataset, q_values_dataset, train_size=0.75, test_size=0.25)

    q_values_dts, results = [], {'training': {}, 'validation': {}}
    for action in range(q_values_dataset.shape[1]):
        q_values_dt = DecisionTreeRegressor(max_depth=max_depth).fit(training_ram_obs, training_q_values[:, action])
        q_values_dts.append(q_values_dt)

        results['training'][f'action {action} score'] = q_values_dt.score(training_ram_obs,
                                                                          training_q_values[:, action])
        results['validation'][f'action {action} socre'] = q_values_dt.score(validation_ram_obs,
                                                                            validation_q_values[:, action])

    results['training']['accuracy'] = accuracy_score(
        onp.argmax([q_values_dt.predict(training_ram_obs) for q_values_dt in q_values_dts], axis=0),
        onp.argmax(training_q_values, axis=1))
    results['validation']['accuracy'] = accuracy_score(
        onp.argmax([q_values_dt.predict(validation_ram_obs) for q_values_dt in q_values_dts], axis=0),
        onp.argmax(validation_q_values, axis=1))

    if env_name:
        total_rewards, total_steps = decision_tree_performance(
            env_name, functools.partial(q_value_dt_policy, decision_trees=q_values_dts), evaluation_seeds)
        results['environment-rewards'], results['environment-steps'] = list(total_rewards), list(total_steps)

    return q_values_dts, results


def train_ensemble_q_value_dt(ram_obs_dataset: onp.ndarray, q_values_dataset: onp.ndarray, max_depth: int = None,
                              env_name: str = None) -> Tuple[List[HistGradientBoostingRegressor],
                                                             List[DecisionTreeRegressor], Dict[str, Any]]:
    """
    Trains an action decision tree and ensemble decision tree policy using sklearn HistGradientBoostingClassifier

    :param ram_obs_dataset: The RAM observation dataset
    :param q_values_dataset: The q-values dataset
    :param max_depth: The max depth of the decision tree
    :param env_name: The environment name
    :return: The decision tree along with the training, validation and environment results
    """
    training_ram_obs, validation_ram_obs, training_q_values, validation_q_values = \
        train_test_split(ram_obs_dataset, q_values_dataset, train_size=0.75, test_size=0.25)

    ensemble_q_values_dts, ensemble_results = [], {'training': {}, 'validation': {}}
    for action in range(q_values_dataset.shape[1]):
        ensemble_q_values_dt: HistGradientBoostingRegressor = HistGradientBoostingRegressor()
        ensemble_q_values_dt.fit(training_ram_obs, training_q_values[:, action])

        ensemble_q_values_dts.append(ensemble_q_values_dt)
        ensemble_results['training'][f'action {action} score'] = \
            ensemble_q_values_dt.score(training_ram_obs, training_q_values[:, action])
        ensemble_results['validation'][f'action {action} score'] = \
            ensemble_q_values_dt.score(validation_ram_obs, validation_q_values[:, action])

    ensemble_results['training']['accuracy'] = accuracy_score(
        onp.argmax([q_values_dt.predict(training_ram_obs) for q_values_dt in ensemble_q_values_dts], axis=0),
        onp.argmax(training_q_values, axis=1))
    ensemble_results['validation']['accuracy'] = accuracy_score(
        onp.argmax([q_values_dt.predict(validation_ram_obs) for q_values_dt in ensemble_q_values_dts], axis=0),
        onp.argmax(validation_q_values, axis=1))

    if env_name:
        total_rewards, total_steps = decision_tree_performance(
            env_name, functools.partial(q_value_dt_policy, decision_trees=ensemble_q_values_dts), evaluation_seeds)
        ensemble_results['environment-rewards'] = list(total_rewards)
        ensemble_results['environment-steps'] = list(total_steps)

    ensemble_q_values = onp.array([q_values_dt.predict(ram_obs_dataset)
                                   for q_values_dt in ensemble_q_values_dts]).reshape(q_values_dataset.shape)
    assert ensemble_q_values.shape == q_values_dataset.shape
    q_value_dts, results = train_q_values_dt(ram_obs_dataset, ensemble_q_values, max_depth=max_depth, env_name=env_name)
    results['ensemble'] = ensemble_results

    return ensemble_q_values_dts, q_value_dts, results


def save_decision_tree(filename: str, data: Any):
    """
    Saves a decision tree (or any data that can be pickled) to the filename

    :param filename: The save location
    :param data: The data to save
    """

    create_directory(filename)
    with open(filename + '.param', 'wb') as file:
        pickle.dump(data, file)


def load_decision_tree(filename: str) -> Any:
    """
    Loads a decision tree from pickle

    :param filename: The filename
    :return: The pickled data
    """
    with open(filename + '.param', 'rb') as file:
        return pickle.load(file)
