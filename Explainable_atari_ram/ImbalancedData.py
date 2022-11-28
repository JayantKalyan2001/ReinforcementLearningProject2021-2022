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
from collections import Counter
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
print(Counter(actionDs))