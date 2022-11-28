"""
Evaluates the performance of the decision trees
"""

import argparse
import json
import os
from pprint import pprint
from typing import Sequence, Any, Dict, Callable

import numpy as np

from dataset import *
from explain import get_env_feature_labels
from train import train_action_dt, save_decision_tree, train_ensemble_action_dt


def _save_results(tree_name: str, tree: Any, tree_results: Dict[str, Any], env_name: str,
                  results: Dict[str, Dict[str, Any]], save_folder: str):
    pprint((tree_name, tree_results))
    results[tree_name] = tree_results
    save_decision_tree(f'{save_folder}/{env_name}/{tree_name}', tree)


def _q_value_distribution(ram_obs, q_values, actions, distribution_percentile: float = 0.05):
    valid_distribution = np.percentile(np.std(q_values, axis=1), distribution_percentile) < np.std(q_values, axis=1)
    print("valid dist",valid_distribution)
    valid_distribution=onp.asarray(valid_distribution)
    print(valid_distribution.shape)
    valid_distribution=valid_distribution.astype(int)
    print(type(valid_distribution[0]))
   # for v in valid_distribution:
    #    print("valid dist element",v)
    return ram_obs[valid_distribution], q_values[valid_distribution], actions[valid_distribution]


def evaluate_hyperparameters(save_folder_fn: Callable[[int, int, str, float], str], envs: Sequence[str] = None,
                             dataset_sizes: Sequence[int] = (25_000,), max_depths: Sequence[int] = (None,),
                             agents: Sequence[str] = ('dqn',), distribution_percentiles: Sequence[float] = (0,),
                             network_root_folder: str = 'jax-models',
                             dataset_root_folder: str = 'datasets'):
    """
    Evaluates the hyperparameters for a set of possible dataset sizes, max depths and agents to the save_folder

    :param save_folder_fn: The save folder of the results passing the dataset size, max depth, agent name
        and distribution percentile
    :param envs: Sequence of envs to run through, if none, try all for the current agent
    :param dataset_sizes: Sequence of dataset sizes
    :param max_depths: Sequence of max depths
    :param agents: Sequence of agent names
    :param distribution_percentiles: Sequence of percentiles for the minimum standard deviations for training q-values
    :param network_root_folder: The network root folder
    :param dataset_root_folder: The dataset root folder
    """
    print(f'Starting to evaluate hyperparameters with sizes: {dataset_sizes}, depths: {max_depths}, agents: {agents}')

    for dataset_size in dataset_sizes:
        for max_depth in max_depths:
            for agent_name in agents:
                for dist_percentile in distribution_percentiles:
                    for env_name in (envs if envs else os.listdir(f'{network_root_folder}/{agent_name}')):
                        save_folder = save_folder_fn(dataset_size, max_depth, agent_name, dist_percentile,env_name)
                        create_directory(save_folder)
                        print(f'Save folder: {save_folder}')
                        results = {'hyperparameters': {'dataset-size': dataset_size, 'max-depth': max_depth,
                                                       'agent-name': agent_name, 'dist-percentile': dist_percentile}}
                        print(results['hyperparameters'])
                        print("dataset root folder: ",dataset_root_folder)
                        if not os.path.exists(f'{dataset_root_folder}/dataset-{dataset_size}/'
                                              f'{agent_name}-{env_name}.npz'):
                            print(f'Generating dataset with size: {dataset_size}, agent: {agent_name}, env: {env_name}')
                            generate_dataset(
                                agent_name=agent_name, env_name=env_name, dataset_size=dataset_size,
                                save_folder=f'{dataset_root_folder}/dataset-{dataset_size}/{agent_name}-{env_name}',
                                network_root_folder=network_root_folder)
                            print("created dataset of ",dataset_size, " for environment ",env_name)
                        # Loads the dataset
                       # ram_obs_dataset, q_value_dataset, action_dataset, metadata = \
                           # load_dataset(f'{dataset_root_folder}/dataset-{dataset_size}/{agent_name}-{env_name}')
                        #print('Dataset metadata:', metadata)

                        ram_obs_dataset=load_ram_obs_from_dataset(save_folder)
                        q_value_dataset=load_q_values_from_dataset(save_folder)
                        action_dataset=load_discrete_actions_from_dataset(save_folder)

                        ram_obs_dataset, _, action_dataset = _q_value_distribution(ram_obs_dataset, q_value_dataset,
                                                                                   action_dataset, dist_percentile)
                        print(f'Reduced dataset size: {len(ram_obs_dataset)} from {dataset_size}')
                        # Train action decision tree
                        action_dt, action_dt_results = train_action_dt(ram_obs_dataset, action_dataset,
                                                                       env_name=env_name, max_depth=max_depth)
                        _save_results('action-dt', action_dt, action_dt_results, env_name, results, save_folder)

                        # Train ensemble action decision tree
                        ensemble_action_dt, action_dt, ensemble_action_dt_results = train_ensemble_action_dt(
                            ram_obs_dataset, action_dataset, env_name=env_name, max_depth=max_depth)
                        _save_results('ensemble-action-dt', (ensemble_action_dt, action_dt), ensemble_action_dt_results,
                                      env_name, results, save_folder)

                        """
                        # Trains q-value decision tree
                        q_values_dts, q_values_dts_results = train_q_values_dt(ram_obs_dataset, q_values_dataset,
                                                                               env_name=env_name, max_depth=max_depth)
                        _save_results('q-values-dt', q_values_dts, q_values_dts_results,
                                      hyperparameter_name, env_name, results, save_folder)
    
                        # Trains ensemble q-value decision tree
                        ensemble_q_values_dt, q_values_dt, ensemble_q_values_dt_results = train_ensemble_q_value_dt(
                            ram_obs_dataset, q_values_dataset, env_name=env_name, max_depth=max_depth)
                        _save_results('ensemble-q-values-dt', (ensemble_q_values_dt, q_values_dt),
                                      ensemble_q_values_dt_results, hyperparameter_name, env_name, results, save_folder)
                        """

                        print(f'Saving results: {results}')
                        with open(f'{save_folder}/{env_name}/results.json', 'w') as file:
                            json.dump(results, file)


def eval_preprocessing_labels(save_folder: str = 'dt-models/preprocessing', network_root_folder: str = 'jax-models',
                              agent_name: str = 'dqn', envs: Sequence[str] = None, dataset_size: int = 50_000,
                              max_depth: int = None, dist_percentile: int = 0, dataset_root_folder: str = 'datasets'):
    create_directory(save_folder)
    print(f'Save folder: {save_folder}')

    for env_name in (envs if envs else os.listdir(f'{network_root_folder}/{agent_name}')):
        results = {'hyperparameters': {'dataset-size': dataset_size, 'max-depth': max_depth,
                                       'agent-name': agent_name, 'dist-percentile': dist_percentile}}
        print(results['hyperparameters'])
        if not os.path.exists(f'{dataset_root_folder}/dataset-{dataset_size}/'
                              f'{agent_name}-{env_name}.npz'):
            print(f'Generating dataset with size: {dataset_size}, agent: {agent_name}, env: {env_name}')
            generate_save_dataset(
                agent_name=agent_name, env_name=env_name, dataset_size=dataset_size,
                save_filename=f'{dataset_root_folder}/dataset-{dataset_size}/{agent_name}-{env_name}',
                network_root_folder=network_root_folder)

        # Loads the dataset
        ram_obs_dataset, q_value_dataset, action_dataset, metadata = \
            load_dataset(f'{dataset_root_folder}/dataset-{dataset_size}/{agent_name}-{env_name}')
        print('Dataset metadata:', metadata)

        ram_obs_dataset, _, action_dataset = _q_value_distribution(ram_obs_dataset, q_value_dataset,
                                                                   action_dataset, dist_percentile)
        print(f'Reduced dataset size: {len(ram_obs_dataset)} from {dataset_size}')

        # Train action decision tree
        _, has_feature_label = get_env_feature_labels(env_name)
        print('has feature label: ', has_feature_label)
        preprocessed_ram_obs_dataset = ram_obs_dataset[:, has_feature_label]
        action_dt, action_dt_results = train_action_dt(preprocessed_ram_obs_dataset, action_dataset,
                                                       env_name=env_name, max_depth=max_depth)
        _save_results('preprocess-action-dt', action_dt, action_dt_results, env_name, results, save_folder)

        # Train ensemble action decision tree
        ensemble_action_dt, action_dt, ensemble_action_dt_results = train_ensemble_action_dt(
            preprocessed_ram_obs_dataset, action_dataset, env_name=env_name, max_depth=max_depth)
        _save_results('preprocess-ensemble-action-dt', (ensemble_action_dt, action_dt), ensemble_action_dt_results,
                      env_name, results, save_folder)

        action_dt, action_dt_results = train_action_dt(ram_obs_dataset, action_dataset,
                                                       env_name=env_name, max_depth=max_depth)
        _save_results('action-dt', action_dt, action_dt_results, env_name, results, save_folder)

        # Train ensemble action decision tree
        ensemble_action_dt, action_dt, ensemble_action_dt_results = train_ensemble_action_dt(
            ram_obs_dataset, action_dataset, env_name=env_name, max_depth=max_depth)
        _save_results('ensemble-action-dt', (ensemble_action_dt, action_dt), ensemble_action_dt_results,
                      env_name, results, save_folder)

        """
        # Trains q-value decision tree
        q_values_dts, q_values_dts_results = train_q_values_dt(ram_obs_dataset, q_values_dataset,
                                                               env_name=env_name, max_depth=max_depth)
        _save_results('q-values-dt', q_values_dts, q_values_dts_results,
                      hyperparameter_name, env_name, results, save_folder)

        # Trains ensemble q-value decision tree
        ensemble_q_values_dt, q_values_dt, ensemble_q_values_dt_results = train_ensemble_q_value_dt(
            ram_obs_dataset, q_values_dataset, env_name=env_name, max_depth=max_depth)
        _save_results('ensemble-q-values-dt', (ensemble_q_values_dt, q_values_dt),
                      ensemble_q_values_dt_results, hyperparameter_name, env_name, results, save_folder)
        """

        print(f'Saving results: {results}')
        with open(f'{save_folder}/{env_name}/results.json', 'w') as file:
            json.dump(results, file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--op')  # ['all-envs', 'max-depth', 'dataset-size', 'training-agent']

    args = arg_parser.parse_args()

    if args.op == 'all-envs':
        evaluate_hyperparameters(lambda _, __, ___, ____: 'dt-models/all-envs',
                                 dataset_sizes=(50_000,), agents=('dqn_adam_mse',))
    elif args.op == 'dataset-size':
        evaluate_hyperparameters(lambda size, _, __, ___: f'dt-models/dataset-size/size-{size}',
                                 dataset_sizes=(1_000, 10_000, 50_000, 100_000))
    elif args.op == 'max-depth':
        evaluate_hyperparameters(lambda _, depth, __, ___: f'dt-models/max-depth/depth-{depth}',
                                 max_depths=(4, 7, 10, None))
    elif args.op == 'training-agent':
        evaluate_hyperparameters(lambda _, __, agent, ___: f'dt-models/training-agent/agent-{agent}',
                                 agents=('dqn', 'c51', 'implicit_quantile', 'quantile', 'rainbow'))
    elif args.op == 'q-value-dist':
        evaluate_hyperparameters(lambda _, __, ___, dist: f'dt-models/q-value-distribution/distribution-{dist}',
                                 distribution_percentiles=(0, 20, 40, 60, 80), dataset_sizes=(100_000,))
    elif args.op == 'breakout-env':
        evaluate_hyperparameters(lambda size, depth, agent, dist: f'dt-models/breakout-env/size-{size}/depth-{depth}/'
                                                                  f'agent-{agent}/dist-{dist}',
                                 dataset_sizes=(50_000, 100_000), max_depths=(10, None),
                                 agents=('dqn', 'c51', 'implicit_quantile', 'quantile', 'rainbow'),
                                 distribution_percentiles=(60, 80))
    elif args.op == 'preprocessing':
        eval_preprocessing_labels()
    else:
        raise Exception('unknown operation:', args.op)
