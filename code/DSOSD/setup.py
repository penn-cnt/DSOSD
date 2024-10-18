from setuptools import setup, find_packages

setup(
    name='DSOSD',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'scipy', 'pandas', 'torch'],  # Dependencies
    description='A package for deploying Neural Dynamic Divergence models',
)