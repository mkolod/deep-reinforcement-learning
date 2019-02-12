[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control - Reacher Environment Solved Using DDPG

### Introduction

This is the directory containing the code to solve Udacity's Deep Reinforcement Learning Nanodegree Project 2 (Continuous Control). Specifically, I trained a [DDPG](https://arxiv.org/pdf/1509.02971.pdf) agent to solve Unity ML Agents' [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 

![Trained Agent][image1]

### Environment Details


In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location, where the goal is keeping the arm in contact with the moving ball. In the above animation, the ball changes color from blue to green when the agent stays in contact with the ball. The goal of the agent is to maintain its position at the target location for as many time steps as possible. 

There are 2 versions of the Unity environment, one with a single environment and one with 20 environments, which can be used by 20 agents.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved if the agent reaches a score of +30.0 in 100 consecutive episodes. For 20 agents, one needs to get the same score over 100 consecutive episodes, but averaged over all agents for each episode.

### Getting Started

We will use the conda environment to ensure that the packages we are installing won't cause conflicts with the default system Python environment.

The following instructions presume that you have **conda** or **Anaconda** (conda plus an optimized Python build and pre-installed data science packages) installed. For more information, see [here](https://conda.io/docs/user-guide/install/index.html) for conda and [here](https://www.anaconda.com/download/) for Anaconda.

1) Set up conda environment

```
conda create --name drlnd python=3.6
source activate drlnd
```

2) Clone this git repository

```git clone https://github.com/mkolod/deep-reinforcement-learning.git```

3) Install required Python packages

```
cd deep-reinforcement-learning
export REPO_ROOT=`pwd`
cd python
pip install .
```

4) Create an IPython kernel

```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5) Download Unity agent and move it to the correct path

The download URLs will depend on your system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

```
cd $REPO_ROOT/p2_continuous-control
wget $URL
unzip Reacher_zip_file.zip
```

6) Start Jupyter notebook and run the code

```
jupyter notebook Continuous_Control.ipynb
```

   

### Instructions

Evaluate the cells in [Continuous_Control.ipynb](./Continuous_Control.ipynb) to see the agent train and perform inference. Note that the notebook already contains the results of a successful session. The agent can be found in [ddpg_agent.py](./ddpg_agent.py), and the model can be found in [model.py](./model.py). 
