"""
Generates the datasets used by the decision trees
"""

import functools
import json
import os
from collections import namedtuple
from re import S
from typing import Dict, Any
from typing import List, Sequence, Callable, Optional, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import supersuit
from flax import linen as nn
from tqdm import tqdm
from tqdm.auto import tqdm

from explainable_atari_ram import create_directory
from explainable_atari_ram.dopamine_setup import get_network_def, load_network_params

DatasetTrajectory = namedtuple('DatasetTrajectory', ['length', 'start', 'end'])


def select_action(obs: onp.ndarray, network_def: nn.Module, network_params: flax.core.FrozenDict,
                  network_args: Dict[str, Any], rng: jax.random.PRNGKey, epsilon: float, num_actions: int,
                  num_envs: int) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
    """
    Selects parallel actions from obs with shape (num_envs, 84, 84, 4) and outputs an updated PRNG,
    the actions randomly selected, the observation's q-values and the argmax actions

    :param obs: Observations with shape (num_envs, 84, 84, 4)
    :param network_def: The network definition
    :param network_params: The network params
    :param network_args: The network argument
    :param rng: Pseudo random number generator
    :param epsilon: The probability of a random action
    :param num_actions: The number of actions in the environment
    :param num_envs: The number of environments
    :return: Tuple of update PRNG, the actions randomly selected, observation's q-values and the argmax actions
    """

    @functools.partial(jax.vmap, in_axes=(0, None))
    def network_q_values(state, args):
        """
        Network Q-values for a particular state
        
        :param state: The state to generate q-values
        :param args: The network arguments
        :return: The Q-values for an environment
        """
      #  print("state/obs values in network_q_values",onp.shape(state))
       # print("getting Q values")
        return network_def.apply(network_params, state, **args)
    #print("out of Q values")
    rng, epsilon_rng, action_rng, network_rng = jax.random.split(rng, num=4)
    #print("computed rng")
    if 'rng' in network_args:
        network_args['rng'] = network_rng
        #print("rng is network args")
    #print(onp.shape(obs)," shape before network_q_values")
    #print("printing network arguments")
    #for n in network_args:
     #   print("network args ",n)
    q_values = network_q_values(obs, network_args).q_values
    #print("computed q values")
    actions = jnp.where(jax.random.uniform(epsilon_rng, (num_envs,)) < epsilon,
                        jax.random.randint(action_rng, (num_envs,), minval=0, maxval=num_actions),
                        jnp.argmax(q_values, axis=1))
    #print("actions computed")
    return rng, q_values, actions


def generate_dataset(agent_name: str, env_name: str, dataset_size: int, save_folder: str,
                     num_envs: int = 20, epsilon: float = 0.01, rng_seed: int = 314159,
                     network_root_folder: str = '../jax-model/jax') -> Tuple[int, int, int]:
    """
    Generates a training dataset using an agent for an environment, actions are selected randomly
        (but not saved to the dataset) epsilon percent of the time

    :param dataset_size: The size of the dataset generate
    :param agent_name: Name of the agent
    :param env_name: The environment name
    :param save_folder: The save folder
    :param num_envs: The number of environments run in parallel
    :param epsilon: The chance of a random action taken
    :param rng_seed: The pseudo random number generator seed
    :param network_root_folder: The root folder for the network params
    :return: returns the episodes_run, steps_taken and saved_steps
    """
    create_directory(save_folder)

    # Image obs, q-values (actions), reward, ram obs for each environment to track transitions
    #print("Image obs, q-values (actions), reward, ram obs for each environment to track transitions")
   # print(" ")
    env_datasets = [([], [], [], [], []) for _ in range(num_envs)]

    # Make the network
    #print("Making the network")
   # print(" ")
    num_actions = gym.make(f'{env_name}NoFrameskip-v4').action_space.n
    network_def, network_args = get_network_def(agent_name, num_actions)
    network_params = load_network_params(agent_name, env_name, network_root_folder=network_root_folder)

    # Random number generator
    rng = jax.random.PRNGKey(seed=rng_seed)

    # Initial the environments and the initial observations
    #print("Initial the environments and the initial observations")
   # print(" ")
    envs = [
        supersuit.frame_stack_v1(gym.wrappers.AtariPreprocessing(gym.make(f'{env_name}NoFrameskip-v4')))
        for _ in range(num_envs)
    ]
    preConObs=[env.reset() for env in envs]
   # print(onp.shape(preConObs)," shape in generate_dataset before concatenation")
  #  print(len(preConObs)," length of list")
   # obs = onp.concatenate([env.reset() for env in envs], axis=0)
    obs=onp.stack(preConObs)
  #  print(len(obs)," length of observation")
 #   print(onp.shape(obs)," shape in generate_dataset")
    # Generate a progress bar as the process can take a while depending on the dataset_size
    progress_bar, progress_bar_update_freq = tqdm(total=dataset_size * 2), dataset_size // 10
    # Save the total steps taken, the number of steps saved so far and
    #   the total number of trajectories / episodes finished
    steps_taken, saved_steps, episodes_run = 0, 0, 0
   # print("Save the total steps taken, the number of steps saved so far and the total number of trajectories / episodes finished")
   # print(" ")
    while saved_steps < dataset_size:
        # Select the actions for the observations
        #print("in loop")
        #print(" ")
        #print(onp.shape(obs)," shape in generate_dataset before select_action is called")
        rng, q_values, actions = select_action(obs, network_def, network_params, network_args,
                                               rng, epsilon, num_actions, num_envs)
        next_obs = onp.empty((num_envs,) + envs[0].observation_space.shape, dtype=envs[0].observation_space.dtype)
        #print("Computed next obs")
        #print(" ")
        # Loop over all the environments
        #print("envs shape",envs.shape)
        for env_num, env in enumerate(envs):
            next_obs[env_num], reward, done, _ = env.step(actions[env_num])
            steps_taken += 1
           # print(onp.shape(obs)," shape in generate_dataset just before new obs")
            # Save the environment information
            env_datasets[env_num][0].append(obs[env_num])
            env_datasets[env_num][1].append(q_values[env_num])
            env_datasets[env_num][2].append(actions[env_num])
            env_datasets[env_num][3].append(reward)
            env_datasets[env_num][4].append(env.unwrapped.ale.getRAM())

            # Update the progress bar every progress_bar_update_freq
            if steps_taken % progress_bar_update_freq == 0:
                progress_bar.update(progress_bar_update_freq)

            # If the environment terminates then save the trajectory
            if done:
                with open(f'{save_folder}/trajectory-{episodes_run}.npz', 'wb') as file:
                    _obs, _q_values, _actions, _reward, _ram_obs = env_datasets[env_num]

                    onp.savez_compressed(file, obs=onp.array(_obs), q_values=onp.array(_q_values),
                                         actions=onp.array(_actions), rewards=onp.array(_reward),
                                         ram_obs=onp.array(_ram_obs), length=len(_obs))

                # update the saved_steps and episodes_runs then reset the environment dataset
                saved_steps += len(_obs)
                episodes_run += 1
                env_datasets[env_num] = ([], [], [], [], [])

                # Set the next obs with the reset of the environment
                next_obs[env_num] = env.reset()

            # Update the observations with the new observation
        obs = next_obs
       # print(onp.shape(obs)," shape in generate_dataset after new obs")
    # Close the progress bar
    progress_bar.close()

    # When the dataset is generated then save the metadata
    with open(f'{save_folder}/metadata.json', 'w') as file:
        json.dump({'agent-name': agent_name, 'env-name': env_name, 'dataset-size': dataset_size, 'num-envs': num_envs,
                   'epsilon': epsilon, 'episodes-run': episodes_run, 'steps-taken': steps_taken,
                   'saved_steps': saved_steps}, file)

    # Return the metadata
    return episodes_run, steps_taken, saved_steps


def _load_dataset_prop(dataset_folder: str, prop: str, dtype: str = None,
                       progress_bar: Optional[Callable[[Sequence], Sequence]] = None) -> onp.ndarray:
    #print(dataset_folder," ", prop)
    if progress_bar is None:
        def _progress_bar(x): return x

        progress_bar = _progress_bar
    dataset = []
    zeroDimArray = False
  #  tempFile =''
    for filename in progress_bar(os.listdir(dataset_folder)):
        if 'trajectory' in filename:
           # print(filename)
           # print(dataset_folder)
           # print(filename)
            with onp.load(f'{dataset_folder}/{filename}', allow_pickle=True) as file_data:
               # print(filename," loaded")
                if dtype is not None:
                  #  print(filename," is not None prop")
                   # print("first element of shape tuple",onp.shape(file_data[prop].astype(dtype))[0])
                    if (onp.shape(file_data[prop].astype(dtype))[0]>1) or (onp.shape(file_data[prop].astype(dtype))[0] is not None):
                      #  print(filename," not zero or 1 dim ",onp.shape(file_data[prop].astype(dtype)))
                      #  dataset=onp.concatenate((dataset,file_data[prop].astype(dtype)),axis=0)
                        for f in file_data[prop].astype(dtype):
                            dataset.append(f)
                    #for f in file_data[prop].astype(dtype):
                           # print(filename,prop,f) 
                    else:
                        
                        zeroDimArray=True
                        
                        return file_data[prop].astype(dtype)[0]

                else:
                  #  print("first element of shape tuple",onp.shape(file_data[prop])[0])
                    if (onp.shape(file_data[prop])[0]>1) or (onp.shape(file_data[prop])[0] is not None) :
                       # print(filename," not zero or 1 dim ",onp.shape(file_data[prop]))
                        #dataset=onp.concatenate((dataset,file_data[prop]),axis=0)
                        for f in file_data[prop]:
                            dataset.append(f)
                   # for f in file_data[prop]:
                    #        print(filename,f)  
                    else:
                       zeroDimArray=True
                       
                       return file_data[prop][0]
   # print (prop,"finished loop")
   # print (onp.shape(dataset))
  #  print(" ")
  #  print("shape of dataset",onp.shape(dataset)," prop:", prop, "filename:", tempFile)
 #   print(dataset)
   # print(dataset)
    if not zeroDimArray:
        #dataset=onp.array(dataset)
        return dataset


load_atari_obs_from_dataset = functools.partial(_load_dataset_prop, prop='obs', dtype=onp.uint8, progress_bar=tqdm)
load_ram_obs_from_dataset = functools.partial(_load_dataset_prop, prop='ram_obs', dtype=onp.uint8)
load_q_values_from_dataset = functools.partial(_load_dataset_prop, prop='q_values')
load_discrete_actions_from_dataset = functools.partial(_load_dataset_prop, prop='actions', dtype=onp.int32)
load_rewards_from_dataset = functools.partial(_load_dataset_prop, prop='rewards')


def load_state_values_from_dataset(dataset_folder: str) -> onp.ndarray:
    """
    Loads the q-values from the dataset and then normalises the values

    :param dataset_folder: the dataset folders
    :return: normalised q-values for the dataset
    """
    q_value_dataset = load_q_values_from_dataset(dataset_folder)
    return onp.max(q_value_dataset, axis=1)


def load_trajectories_from_dataset(dataset_folder: str) -> List[DatasetTrajectory]:
    """
    Loads the trajectories with the trajectory size, start and end from the dataset

    :param dataset_folder: the dataset folder
    :return: List of trajectory sizes, the start and end position in the trajectory
    """
    trajectories, pos = [], 0
    for filename in os.listdir(dataset_folder):
        if 'trajectory' in filename:
            length = onp.load(f'{dataset_folder}/{filename}', allow_pickle=True)['length']
            trajectories.append(DatasetTrajectory(length, pos, pos + length))

            pos += length

    return trajectories


def dataset_to_trajectories(dataset: onp.ndarray, trajectories: List[DatasetTrajectory]) -> List[onp.ndarray]:
    """
    Converts the dataset to a list of trajectories

    :param dataset: the dataset with the trajectories concatenated
    :param trajectories: the trajectories from the dataset
    :return: List of trajectories
    """
    return [dataset[trajectory.start: trajectory.end] for trajectory in trajectories]
