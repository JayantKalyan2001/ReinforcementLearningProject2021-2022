"""
Functions to load the dopamine-rl models and altered version of ImplicitQuantileNetwork that have the output
    network type (ImplicitQuantileNetworkType) have a q-value output
"""

import collections
import pickle
from typing import Dict, Any, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as onp
from dopamine.jax.networks import *
from flax import linen as nn
from flax.training import checkpoints

# Source for this class is dopamine.jax.networks
# Modified to return the q_values not just the quantile_values and quantiles
ImplicitQuantileNetworkType = collections.namedtuple('iqn_network', ['q_values', 'quantile_values', 'quantiles'])



class ImplicitQuantileNetwork(nn.Module):
    """
    A copy of the Implicit Quantile Network from google dopamine.jax.networks,
        except the output is q-values not quantiles, ImplicitQuantileNetworkType

    The Implicit Quantile Network (Dabney et al., 2018)
    """
    num_actions: int
    quantile_embedding_dim: int

    @nn.compact
    def __call__(self, x, num_quantiles, rng):
        initializer = nn.initializers.variance_scaling(scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform')

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # flatten
        state_vector_length = x.shape[-1]
        state_net_tiled = jnp.tile(x, [num_quantiles, 1])
        quantiles_shape = [num_quantiles, 1]
        quantiles = jax.random.uniform(rng, shape=quantiles_shape)
        quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
        quantile_net = (jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32) * onp.pi * quantile_net)
        quantile_net = jnp.cos(quantile_net)
        quantile_net = nn.Dense(features=state_vector_length, kernel_init=initializer)(quantile_net)
        quantile_net = nn.relu(quantile_net)
        x = state_net_tiled * quantile_net
        x = nn.Dense(features=512, kernel_init=initializer)(x)
        x = nn.relu(x)
        quantile_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
        q_values = jnp.mean(quantile_values, axis=0)
        return ImplicitQuantileNetworkType(q_values, quantile_values, quantiles)


def get_network_def(agent_name: str, num_actions: int) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads the network definition and any additional information required for running the model

    :param agent_name: The agent name (c51, dqn, dqn_adam_mse, implicit_quantile, quantile, rainbow)
    :param num_actions: The number of actions in the environment
    :return: flax neural network module and dictionary for calling .apply
    """
    if agent_name == 'c51' or agent_name == 'rainbow':
        return RainbowNetwork(num_actions=num_actions, num_atoms=51), {'support': jnp.linspace(-10, 10, 51)}
    elif agent_name == 'dqn' or agent_name == 'dqn_adam_mse':
        print("chosen network def for dqn")
        return NatureDQNNetwork(num_actions=num_actions), {}
    elif agent_name == 'implicit_quantile':
        return ImplicitQuantileNetwork(num_actions=num_actions, quantile_embedding_dim=64), \
               {'num_quantiles': 32, 'rng': None}
    elif agent_name == 'quantile':
        return QuantileNetwork(num_actions=num_actions, num_atoms=200), {}
    else:
        raise Exception(f'Unknown agent type: {agent_name}')


def load_network_params(agent_name: str, env_name: str,
                        network_root_folder: str = 'jax-models') -> flax.core.FrozenDict:
    """
    Loads the network params of dopamine pre-trained JAX agents
    (we select the second agent due to bugs in the dqn/SpaceInvaders/1 agent
     https://github.com/google/dopamine/issues/181 now fixed)

    :param agent_name: The agent name (c51, dqn, dqn_adam_mse, implicit_quantile, quantile, rainbow)
    :param env_name: The environment name, must be an atari environment name
    :param network_root_folder: The root folder
    :return: Flax FrozenDict
    """
    with open(f'{network_root_folder}/{agent_name}/{env_name}/2/ckpt.199', 'rb') as file:
        param_data = pickle.load(file)
        network_params = flax.core.FrozenDict({
            'params': checkpoints.convert_pre_linen(
                param_data['online_params']).unfreeze()
        })
    return network_params
