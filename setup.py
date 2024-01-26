#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='multiagent_envs',
    version='0.0.1',
    description='Multi-agent Environments',
    author='Yuchen Xiao',
    author_email='xiao.yuch@husky.neu.edu',
    url='https://github.ccs.neu.edu/ycx424/multiagent-envs',
    packages=['marl_envs'],
    package_dir={'': 'src'},
    license='MIT',
)