from setuptools import setup, find_packages

setup(
    name='NDD',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'scipy', 'pandas', 'torch'],  # Dependencies
    description='A package for deploying Neural Dynamic Divergence models',
)