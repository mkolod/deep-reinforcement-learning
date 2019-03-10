[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition Using Multi-Agent Deep Deterministic Policy Gradient (MAGGPG)

### Project Details

This project solves Udacity Deep Reinforcement Learning Nanodegree Program's Project 3 - Collaboration and Competition. The environment to be solved is called Tennis and comes from [Unity ML Agents'](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment collection.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  **score**  for each episode.

The environment is considered solved, when the average (over 100 episodes) of those  **scores**  is at least +0.5.

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

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    

6. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/`

To download the binary and place it in the correct location, do the following:
```
cd $REPO_ROOT/p2_collab-compet
wget $URL
unzip Tennis_zip_file.zip
```

6) Start Jupyter notebook and run the code

```
jupyter notebook Tennis.ipynb
```

   

### Instructions

Evaluate the cells in`Tennis.ipynb` to see the agent train and perform inference. Note that the notebook already contains the results of a successful session.
