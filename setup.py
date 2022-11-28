"""
Setup script for the explainable RAM observations project
"""

from setuptools import setup

setup(
    name='explainable-atari-ram',
    version='0.1',
    author='Mark Towers',
    author_email='mt5g17@soton.ac.uk',
    python_requires='>=3.7',
    install_requires=['dopamine-rl', 'jax[cpu]', 'tqdm', 'sklearn', 'supersuit', 'pytest', 'pydotplus',
                      'jupyter', 'matplotlib', 'shap', 'gym', 'flax', 'numpy', 'opencv-python',
                      'scipy']
)
