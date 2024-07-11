

Deep RL is a type of Machine Learning where an agent learns how to behave in an environment by performing actions and seeing the results.

A formal definition:
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

# The Reinforcement Learning Framework.
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/sars.jpg)

The agent’s goal is to maximize its cumulative reward, called the expected return.

The reward hypothesis: the central idea of Reinforcement Learning.

Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/obs_space_recap.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/action_space.jpg)

### Rewards and the discounting

To discount the rewards, we proceed like this:

- We define a discount rate called gamma. It must be between 0 and 1. Most of the time between 0.95 and 0.99.
- The larger the gamma, the smaller the discount. This means our agent cares more about the long-term reward.

expected cumulative reward is:
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_4.jpg)

### Type of tasks
A task is an instance of a Reinforcement Learning problem. We can have two types of tasks: episodic and continuing.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/tasks.jpg)

###  The Exploration/Exploitation trade-off

the exploration/exploitation trade-off.

- Exploration is exploring the environment by trying random actions in order to find more information about the environment.
- Exploitation is exploiting known information to maximize the reward.

### Two main approaches for solving RL problems

- The Policy π: the agent’s brain
The Policy π is the brain of our Agent, it’s the function that tells us what action to take given the state we are in. So it defines the agent’s behavior at a given time.

This Policy is the function we want to learn, our goal is to find the optimal policy π*, the policy that maximizes expected return when the agent acts according to it. We find this π* through training.

There are two approaches to train our agent to find this optimal policy π*:
- Directly, by teaching the agent to learn which action to take, given the current state: Policy-Based Methods.
- Indirectly, teach the agent to learn which state is more valuable and then take the action that leads to the more valuable states: Value-Based Methods.

#### Policy-Based Methods:
In Policy-Based methods, we learn a policy function directly.

We have two types of policies:

- Deterministic: a policy at a given state will always return the same action.
- Stochastic: outputs a probability distribution over actions.
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_1.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_2.jpg)

#### Value-based methods
In value-based methods, instead of learning a policy function, we learn a value function that maps a state to the expected value of being at that state.

The value of a state is the expected discounted return the agent can get if it starts in that state, and then acts according to our policy.

“Act according to our policy” just means that our policy is “going to the state with the highest value”.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/vbm_1.jpg)

## Summary

That was a lot of information! Let’s summarize:

- Reinforcement Learning is a computational approach of learning from actions. We build an agent that learns from the environment by interacting with it through trial and error and receiving rewards (negative or positive) as feedback.

- The goal of any RL agent is to maximize its expected cumulative reward (also called expected return) because RL is based on the reward hypothesis, which is that all goals can be described as the maximization of the expected cumulative reward.

- The RL process is a loop that outputs a sequence of state, action, reward, and next state.

- To calculate the expected cumulative reward (expected return), we discount the rewards: the rewards that come sooner (at the beginning of the game) are more probable to happen since they are more predictable than the long-term future reward.

- To solve an RL problem, you want to find an optimal policy. The policy is the “brain” of your agent, which will tell us what action to take given a state. The optimal policy is the one that gives you the actions that maximize the expected return.

- There are two ways to find your optimal policy:
  - By training your policy directly: policy-based methods.
  - By training a value function that tells us the expected return the agent will get at each state and use this function to define our policy: value-based methods.

- Finally, we speak about Deep RL because we introduce deep neural networks to estimate the action to take (policy-based) or to estimate the value of a state (value-based) hence the name “deep”.


## Glossary

### Agent
An agent learns to make decisions by trial and error, with rewards and punishments from the surroundings.

### Environment
An environment is a simulated world where an agent can learn by interacting with it.

### Markov Property
It implies that the action taken by our agent is conditional solely on the present state and independent of the past states and actions.

### Observations/State
- **State:** Complete description of the state of the world.
- **Observation:** Partial description of the state of the environment/world.

### Actions
- **Discrete Actions:** Finite number of actions, such as left, right, up, and down.
- **Continuous Actions:** Infinite possibility of actions; for example, in the case of self-driving cars, the driving scenario has an infinite possibility of actions occurring.

### Rewards and Discounting
- **Rewards:** Fundamental factor in RL. Tells the agent whether the action taken is good/bad. RL algorithms are focused on maximizing the cumulative reward.
- **Reward Hypothesis:** RL problems can be formulated as a maximization of (cumulative) return. Discounting is performed because rewards obtained at the start are more likely to happen as they are more predictable than long-term rewards.

### Tasks
- **Episodic:** Has a starting point and an ending point.
- **Continuous:** Has a starting point but no ending point.

### Exploration v/s Exploitation Trade-Off
- **Exploration:** It’s all about exploring the environment by trying random actions and receiving feedback/returns/rewards from the environment.
- **Exploitation:** It’s about exploiting what we know about the environment to gain maximum rewards.
- **Exploration-Exploitation Trade-Off:** It balances how much we want to explore the environment and how much we want to exploit what we know about the environment.

### Policy
- **Policy:** It is called the agent’s brain. It tells us what action to take, given the state.
- **Optimal Policy:** Policy that maximizes the expected return when an agent acts according to it. It is learned through training.

### Policy-based Methods
An approach to solving RL problems. In this method, the Policy is learned directly. Will map each state to the best corresponding action at that state. Or a probability distribution over the set of possible actions at that state.

### Value-based Methods
Another approach to solving RL problems. Here, instead of training a policy, we train a value function that maps each state to the expected value of being in that state.


### PPO is a combination of:

- Value-based reinforcement learning method: learning an action-value function that will tell us the most valuable action to take given a state and action.
- Policy-based reinforcement learning method: learning a policy that will give us a probability distribution over actions.


## Quiz

### What is Reinforcement Learning?
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

# Additional Readings

- Deep learning
  -  [the FastAI Practical Deep Learning for Coders](https://course.fast.ai/)
- Deep Reinforcement Learning
  - [Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 1, 2 and 3](http://incompleteideas.net/book/RLbook2020.pdf)
  - [Foundations of Deep RL Series, L1 MDPs, Exact Solution Methods, Max-ent RL by Pieter Abbeel](https://www.youtube.com/watch?v=2GwBez0D20A&ab_channel=PieterAbbeel)
  - [Spinning Up RL by OpenAI Part 1: Key concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- Gym
  - [Getting Started With OpenAI Gym: The Basic Building Blocks](https://blog.paperspace.com/getting-started-with-openai-gym/)
  - [Make your own Gym custom environment](https://www.gymlibrary.dev/content/environment_creation/)

## Huggy is a Deep Reinforcement Learning environment made by Hugging Face and based on Puppo the Corgi, a project by the Unity MLAgents team.
After each action Puppo performs, a reward to the agent is given. The reward is comprised of:

![](https://blog-api.unity.com/sites/default/files/styles/focal_crop_ratio_16_9/public/2018/10/Puppo-Reward.png?h=9244b94d&itok=LXNKzhJ1)

- **Orientation Bonus**: We reward Puppo when it is moving towards the target. To do so, we use the Vector3.Dot() method.
- **Time Penalty**: We give a fixed penalty (negative reward) to Puppo at every action. This way, Puppo will learn to get the stick as fast as possible to avoid a heavy time penalty.
- **Rotation Penalty**: We penalize Puppo for trying to spin too much. A real dog would be dizzy if it spins too much. To make it look real, we penalize Puppo when it turns around too fast.
- **Getting to the target Reward**: Most importantly, we reward Puppo for getting to the target.