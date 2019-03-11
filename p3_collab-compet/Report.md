[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

[image2]: ./maddpg_scores.png "Agent Scores"

## Report regarding training of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Agent to Play Unity ML Agents' Tennis Environment

![Trained Agent][image1]

### Problem Statement

This project solves Udacity Deep Reinforcement Learning Nanodegree Program's Project 3 - Collaboration and Competition. The environment to be solved is called Tennis and comes from [Unity ML Agents'](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment collection.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  **score**  for each episode.

The environment is considered solved, when the average (over 100 episodes) of those  **scores**  is at least +0.5.

The environment was solved using Multi-Agent Deep Deterministic Policy Gradient (MADDPG). See the [paper](https://arxiv.org/pdf/1706.02275.pdf) for details.

### Learning Algorithm

#### DDPG

Before we talk about Multi-Agent Deep Deterministic Policy Gradient (MADDPG), let's talk about a single-agent case. Most of this section is copied over from problem 2's Report.md, though the explanation of the multi-agent case is new.

On the surface, [DDPG](https://arxiv.org/pdf/1509.02971.pdf) looks like an actor-critic model, because we have two networks. Strictly speaking though, it's not an actor-critic model, though the authors said that it was. Thus, let's first refresh what an actor-critic model is about. 

Actor-critic models are not only based on two networks, they are based on two different reinforcement learning approaches. The actor network is normally a Monte Carlo-based, value function predicting model. The strength of such a network is that it's unbiased, because entire episodes are rolled out by collecting actual feedback from the environment. Unfortunately, they are also high variance, because many trajectories can pass through the same state. Also, rolling out entire episodes means that the actor is very data-inefficient. The critic is the opposite - it's a policy network, which uses a temporal difference (TD) approach, which means that only one step has to be simulated (the immediate interaction). The discounted future reward is taken from a prediction, but since we then use an estimate rather than a roll-out, a TD approach can have bias, even though it has reduced variance (because there's only one direct interaction with the environment). Combining the no-bias, high-variance actor and the low-bias, low-variance critic, we eventually get the actor which is both lower variance and has some initial guesses to work off of to speed up convergence. Of course, once the model is trained, we discard the critic and just use the actor for inference, so inference is still efficient. Training may require more compute, but the convergence is quicker and more stable, so having the critic help the actor is a net win.

It was worth explaining traditional actor-critic models to compare DDPG against them. In a lot of ways, even though DDPG has two networks, it's not really an actor-critic in the traditional sense. It can be seen as an approximate DQN. The reason is that the "critic" in DDPG is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline. 

The main problem with DQN agents is that it's not obvious how to use them for problems with continuous action spaces. When a DQN outputs the Q values for each state-action pair (or rather, for the current state and each action), it's easy to choose the best action using the argmax over those outputs. However, what would we do if we have a continuous output? We no longer can take the argmax, because it's not a discrete action space anymore - we have to guess what the max should be in a continuous space. 

DDPG solves the continuous action space problem. We use 2 neural networks. Even though DDPG is not really a "true" actor-critic model (because it doesn't combine a Monte Carlo action-value network with a TD learning-based policy network), let's use that terminology for now instead of referring to network 1 and network 2. In DDPG, the actor predicts the optimal policy deterministically. This means the actor gives only the best action, rather than a probability distribution over actions. In other words, the actor predicts argmax<sub>a</sub>Q(s, a). Of course, it's not a discrete argmax because the actions are continuous, it's the action which is somewhere on the real line of continuous values that ends up maximizing Q. Let's call this $\mu(s, \theta_{\mu}$). 
The critic learns to evaluate the optimal value function based on the actor's best chosen action. We can write this as $Q(s, \mu(s, \theta_{\mu}); \theta_Q)$. To rephrase, we use the actor, which is an approximate maximizer, to calculate a new target value to train a new action-value function, which the critic provides.  This does sound like an approximate DQN. Another DQN-style feature is the use of the replay buffer. Another thing which is important to mention is the use of the soft update approach. In DQNs, there's a local network and a target network. The target is normally updated only once in a few hundred of thousand time steps, while the local actor is updated at each time step. The updates are then done to the target by copying the weights at those irregular intervals. The DDPG approach, and indeed one used in other modern architectures, is to make much smaller, but regular updates to the target by linearly interpolating between the target and the local actor. To make sure the updates are small, the weight attached to the contribution of the local actor is $0 <\tau<1$, with the value of $\tau$ being much closer to 0 than to 1, and with the weight attached to the last value of the target being $1-\tau$, which clearly then ends up being much closer to 1 than to 0.

Last but not least, if it wasn't obvious so far, DDPG is an off-policy, model-free algorithm. 
For more details, see the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf).

#### MADDPG

DDPG works great for a single agent, but how can we deal with multiple agents? A multi-agent scenario is challenging compared to a fixed environment because it's non-stationary - if other agents are learning at the same time as our current agent is, then the feedback that our agent gets from the environment can't be relied upon in the long run to give feedback. This, among other things, reduces the benefit of the replay buffer, which holds states, actions, rewards, and next states, but the states are no longer static since they depend on other agents learning. How do we go about addressing non-stationarity of the feedback the agent is getting?

[MADDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) does precisely that. One of the key principles behind this algorithm is that we should only obtain information other than the agent's observation vector during training, because the agent needs to be independent and depend only on its own information at inference time. One candidate for allowing such training-only information is the critic. Since only the agent is needed at inference time and the critic is only used at training time, but only feeding extra information into the critic, we allow for more robust learning while not making the agent dependent on anything except its own observations at inference time. For example, we can concatenate our agent's observations with other agents' observations to allow our agent's critic to learn better, but still only depend on our agent's states for the actor model.

One thing that one can get confused about when trying to understand MADDG and similar models is the notion of a "centralized critic." This need not actually mean that we have N agents and only one critic. For example, let's say that our agent collaborates with some other agents and competes with others. Naturally, it would be useful for each agent to have its own critic. The only "centralized" component here is the information about the state of the environment, or rather the agent's observations, which can include both the state of the environment and the behavior or other agents.

#### Model details

Let's start with a single DDPG agent, and specifically its actor and critic. These are very simple and similar to the ones I had for problem 2 [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p2_continuous-control/model.py). The agent used (code [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/model.py#L16:L55)) is simply a multi-layer perceptron (MLP). It gets the Nx24 state vector, where N is the batch size and there are 24 observations of the environment at any given point. Instead of using leaky ReLU and Kaiming He's initialization like in problem 2, I opted for a simple MLP with ReLU and a basic fan-in initialization. Note though that after the first layer's affine transformation and nonlinearity, I added BatchNorm ([paper](https://arxiv.org/pdf/1502.03167.pdf)). This layer was added in order to minimize issues having to do with outputs of different magnitudes from the previous layer. BatchNorm tends to stabilize training and in some cases reduce overfitting. Since we want the actions to be in a narrow range, in this case between -1 and +1, the output non-linearity is a hyperbolic tangent (tanh). The critic ([code](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/model.py#L58:L94)) is similarly straightforward. The only thing that we need to pay attention to in the critic is that we first need to concatenate the states provided (which includes both our current actor's observations, and other actors' observations) with the actor's actions (see [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/model.py#L87)). This is the coupling between the actor and the critic - the critic takes the environment state and the actor's actions and determines the cumulative reward the agent can expect.

The [replay buffer](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/replay_buffer.py), [Ornstein–Uhlenbeck noise](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/ou_noise.py) and the [soft update](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/utils.py#L3:L13) are the same as before, so I won't discuss them further - for details, see [problem 2](https://github.com/mkolod/deep-reinforcement-learning/tree/master/p2_continuous-control).

Now that we understand the actor and the critic, let's look at the DDPG agent. There is a detailed explanation in the [Report for problem 2](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md), but this is the multi-agent version, so let's both recap and see if anything is different. The [DDPG agent](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L122:L247) is similar to the DDPG agent we saw last time, but because it's a mutli-agent environment, let's point out that it needs to keep track of its own ID. This is needed because the agent is learning from not just its own observation, but other agents' observations as well. We can see this mostly because the agent only needs other agents' states, but it only relies on its own actions and rewards, so we need to select only ones belonging to the current agent (see [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L190:L191)).  As in problem 2, we create both a "local" and a "target" actor and critic (see [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L149:L155)). This is because as in the DQN model, we want to decorrelate the predictions of the discounted returns of future actions from the actions predicted based on current states. For more details as to why it's important, see the [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). Note though that unlike in the DQN paper, we use the soft update to linearly interpolate between the current (local) actor/critic and the target (stable) actor/critic. Once we initialize the local and target actors and critics, we need to ensure that local and target are starting from the same initialization, hence the "hard update," which essentially copies the weights to make sure the local and target are the same for both the actor and the critic (see [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L157:L158)). We also initialize the optimizers for the actor and the critic. I'm using Adam [here](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L160:L161), since it tends to perform well if a good guess for an initial learning rate is unknown. Still, I found that tweaking the learning rate was necessary and that I needed to set it low for both the actor and the critic anyway (1e-4).

The DDPG agent's [act()](https://github.com/mkolod/deep-reinforcement-learning/blob/master/p3_collab-compet/maddpg_agent.py#L174:L182) method is the same as in problem 2 - we get the data, put the actor in inference mode (`self.actor_local.eval()`) and run without recording gradients. We simply call the actor's `forward()` method. Since inference is complete, we need to put the actor back in training mode. The actions need some noise to ease training, so the [Ornstein-Uhlenbeck noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) is applied at this stage. The actions are in range [-1, 1], so we clip them to that value.






### Plot of Rewards

To recall from earlier, in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  **score**  for each episode.

The environment is considered solved, when the average (over 100 episodes) of those  **scores**  is at least +0.5.

Here is the plot of the scores. The scores look pretty small and constant for the first 1,000 episodes, after which they start growing. Since the instantaneous scores (in blue) are very noisy, I added the 100-episode moving average (in orange), which specifically tracks the expected solution, i.e. a mean score of 0.5 over the last 100 episodes.

![Agent Scores][image2]


### Ideas for Future Work

The submission has concrete future ideas for improving the agent's performance.


