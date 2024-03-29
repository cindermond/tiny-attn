import setuptools
from setuptools import setup

setup(
    name='rewriter_bl',
    version='1.0',
    description='rewriter',
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers==4.19.2',
        'datasets',
        'scipy',
        'sklearn',
        'future'
    ]
    )