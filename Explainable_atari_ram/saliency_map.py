"""
Generate a saliency map following Greydanus et al, 2018
"""

from typing import Tuple, Dict, Any

import cv2
import jax.numpy as jnp
import numpy as onp
from flax import linen as nn
from flax.core import FrozenDict
from gym.envs.atari import AtariEnv
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal


def generate_saliency_mask(center, mask_points, radius=25):
    """
    Generates the saliency mask

    :param center: The mask center position
    :param mask_points: Numpy array of mask points
    :param radius: The radius of the mask
    :return: Normalised mask over the mask points at the center (mean) and radius (standard deviation)
    """
    mask = multivariate_normal.pdf(mask_points, center, radius)
    return mask / mask.max()


def generate_saliency_map(obs: onp.ndarray, network_def: nn.Module, network_params: FrozenDict,
                          network_args: Dict[str, Any], patch_width: int = 5) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Generate the saliency map for a particular observation and JAX network definition

    :param obs: Environment observation (84, 84, 4)
    :param network_def: JAX network definition
    :param network_params: JAX network parameters
    :param network_args: JAX network arguments for more complex networks
    :param patch_width: The patch width to improve performance, such that we find the saliency every patch width
    :return: Tuple of the saliency map and the raw saliency values in a 2d array
    """
    assert obs.shape == (84, 84, 4), obs.shape

    original_q_values = network_def.apply(network_params, obs, **network_args).q_values
    saliency_map = onp.zeros((84, 84))
    saliency_values = onp.zeros((84 // patch_width + 1, 84 // patch_width + 1))

    blurred_obs = gaussian_filter(obs, sigma=3)

    mask_xx, mask_yy = onp.meshgrid(onp.arange(84), onp.arange(84))
    mask_points = onp.stack((mask_xx, mask_yy), axis=-1)

    for x_center in onp.arange(0, 84, patch_width):
        for y_center in onp.arange(0, 84, patch_width):
            mask = onp.repeat(generate_saliency_mask((x_center, y_center), mask_points)[:, :, onp.newaxis], 4, axis=2)
            perturbed_obs = obs * (1 - mask) + blurred_obs * mask
            perturbed_q_values = network_def.apply(network_params, perturbed_obs, **network_args).q_values

            saliency = jnp.mean(jnp.square(original_q_values - perturbed_q_values))
            saliency_map += saliency * multivariate_normal.pdf(mask_points, (x_center, y_center), patch_width ** 2)
            saliency_values[x_center // patch_width, y_center // patch_width] = saliency

    return onp.array(255 * saliency_map / onp.max(saliency_map), dtype=onp.uint16), saliency_values


def render_saliency_map(env: AtariEnv, obs: onp.ndarray, network_def: nn.Module, network_params: FrozenDict,
                        network_args: Dict[str, Any], patch_width: int = 5, channel: int = 2):
    """
    Render the saliency map for a particular env and obs for a network

    :param env: Atari environment
    :param obs: Current environment observation (84, 84, 4)
    :param network_def: JAX network definition
    :param network_params: JAX network parameters
    :param network_args: JAX network arguments for more complex networks
    :param patch_width: The patch width to improve performance, such that we find the saliency every patch width
    :param channel: The channel (red, green, blue) in which to add the saliency values with default as 2 (blue)
    :return: (160, 210) image from the env.render rgb array with the saliency map
    """
    render_obs = env.render('rgb_array').astype(onp.uint16)

    saliency_map, _ = generate_saliency_map(obs, network_def, network_params, network_args, patch_width)
    reshaped_saliency_map = cv2.resize(saliency_map, (160, 210))
    render_obs[:, :, channel] += reshaped_saliency_map

    return render_obs.clip(0, 255).astype(onp.uint8)
