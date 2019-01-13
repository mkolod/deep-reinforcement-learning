## Navigation in Unity ML-Agents' Banana Environment

## What is this?

This project implements the training and inference of Udacity's [Navigation environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation), which is a slightly modified version of Unity ML Agents' [Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). The goal is to collect as many yellow bananas as possible, while avoiding blue bananas. Collecting a yellow banana gives a reward of +1, while collecting a blue banana gives a reward of -1. The agent has 4 actions to choose from: move left/right/forward/backward. This is an episodic task. The environment is considered solved if the agent gets an average score of +13 over 100 consecutive episodes. The state space has 37 dimensions. These dimensions represent the ray-based perception of objects around the agent's forward direction, as well as the agent's velocity.


## Installation / Running instructions

We will use the conda environment to ensure that the packages we are installing won't cause conflicts with the default system Python environment.

### How to install the environment:

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

- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

Assume that the environment variable URL contains the URL appropriate for your system.
```
cd $REPO_ROOT/p1_navigation
wget $URL
unzip Banana.app.zip
```

6) Start Jupyter notebook and run the code

```
jupyter notebook Navigation.ipynb
```

## Demo of trained agent

The inference video of the agent trained using the code in [Navigation.ipynb](./Navigation.ipynb) can be found [here](https://youtu.be/DZZhpnpUCPQ)

