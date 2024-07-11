 ## value-based methods

In value-based methods, we learn a value function that maps a state to the expected value of being at that state.

The value of a state is the expected discounted return the agent can get if it starts at that state and then acts according to our policy.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/vbm-1.jpg)

Remember that the goal of an RL agent is to have an optimal policy π*.

#### the difference is:

In policy-based training, the optimal policy (denoted π*) is found by training the policy directly.
In value-based training, finding an optimal value function (denoted Q* or V*) leads to having an optimal policy.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg)

We have two types of value-based functions:

### The state-value function

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/state-value-function-1.jpg)

### The action-value function

In the action-value function, for each state and action pair, the action-value function outputs the expected return if the agent starts in that state, takes that action, and then follows the policy forever after.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/action-state-value-function-1.jpg)

We see that the difference is:

- For the state-value function, we calculate the value of a state $S_t$.
- For the action-value function, we calculate the value of the state-action pair ($S_t$​,$A_t$​) hence the value of taking that action at that state.

## The Bellman Equation: simplify our value estimation

Instead of calculating the expected return for each state or each state-action pair, we can use the Bellman equation.

The Bellman equation is a recursive equation that works like this: instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as:

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman4.jpg)

To recap, the idea of the Bellman equation is that instead of calculating each value as the sum of the expected return, which is a long process, we calculate the value as the sum of immediate reward + the discounted value of the state that follows.

## Monte Carlo vs Temporal Difference Learning
Remember that an RL agent learns by interacting with its environment. The idea is that given the experience and the received reward, the agent will update its value function or policy.

Monte Carlo and Temporal Difference Learning are two different strategies on how to train our value function or our policy function. Both of them use experience to solve the RL problem.

On one hand, Monte Carlo uses an entire episode of experience before learning. On the other hand, Temporal Difference uses only a step ($S_t$,$A_t$,$R_{t+1}$,$S_{t+1}$) to learn.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/monte-carlo-approach.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-3p.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-5p.jpg)

## Temporal Difference Learning: learning at each step

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1.jpg)

![Alt text](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1p.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-3p.jpg)

#### To summarize:

With Monte Carlo, we update the value function from a complete episode, and so we use the actual accurate discounted return of this episode.

With TD Learning, we update the value function from a step, and we replace $G_t$, which we don’t know, with an estimated return called the TD target.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Summary.jpg)

What is the Bellman Equation?

The Bellman equation is a recursive equation that works like this: instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as: The immediate reward + the discounted value of the state that follows

## Q-learning:

is an off-policy value-based method that uses a TD approach to train its action-value function.

Q-Learning is the algorithm we use to train our Q-function, an action-value function that determines the value of being at a particular state and taking a specific action at that state.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function.jpg)

The Q comes from “the Quality” (the value) of that action at that state.

Let’s recap the difference between value and reward:

- The value of a state, or a state-action pair is the expected cumulative reward our agent gets if it starts at this state (or state-action pair) and then acts accordingly to its policy.
- The reward is the feedback I get from the environment after performing an action at a state.

Internally, our Q-function is encoded by a Q-table, a table where each cell corresponds to a state-action pair value.

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg)

The Q-Learning algorithm
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-2.jpg)

- step 1: initialize the Q table
- step 2: choose an action using the epsilon-greedy strategy
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-4.jpg)

The epsilon-greedy strategy is a policy that handles the exploration/exploitation trade-off.

The idea is that, with an initial value of ɛ = 1.0:
- With probability 1 — ɛ : we do exploitation (aka our agent selects the action with the highest state-action pair value).
- With probability ɛ: we do exploration (trying random action).

At the beginning of the training, the probability of doing exploration will be huge since ɛ is very high, so most of the time, we’ll explore. But as the training goes on, and consequently our Q-table gets better and better in its estimations, we progressively reduce the epsilon value since we will need less and less exploration and more exploitation.

- Step 3: Perform action At, get reward Rt+1 and next state St+1
- Step 4: Update Q(St, At)
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-8.jpg)

How do we form the TD target?
- We obtain the reward $R_{t+1}$ after taking the action $A_t$.
-To get the best state-action pair value for the next state, we use a greedy policy to select the next best action.

Note that this is not an epsilon-greedy policy, this will always take the action with the highest state-action value.


Then when the update of this Q-value is done, we start in a new state and select our action using a epsilon-greedy policy again.

### Off-policy vs On-policy

Off-policy: using a different policy for acting (inference) and updating (training).

For instance, with Q-Learning, the epsilon-greedy policy (acting policy), is different from the greedy policy that is used to select the best next-state action value to update our Q-value (updating policy).

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-4.jpg)

# Glossary
## Strategies to Find the Optimal Policy

### Policy-based Methods
Policy-based methods involve training a neural network to select actions based on the current state. The neural network outputs the action for the agent, and adjustments are made based on the environment's feedback, leading to improved actions.

### Value-based Methods
In value-based methods, a value function is trained to output the value of a state or state-action pair, representing the policy. The agent's behavior is determined by the output of the value function. For example, a Greedy Policy may be adopted, where the agent always takes the action with the highest reward.

### Value-based Strategies
1. **State-value Function**
   - Represents the expected return for each state when the agent follows the policy until the end.

2. **Action-value Function**
   - Calculates the expected return for each state-action pair when the agent starts in a state, takes an action, and follows the policy thereafter.

### Epsilon-greedy Strategy
A common strategy in reinforcement learning that balances exploration and exploitation.
- Chooses the action with the highest expected reward with a probability of 1-epsilon.
- Chooses a random action with a probability of epsilon.
- Epsilon is typically decreased over time to shift focus towards exploitation.

### Greedy Strategy
Involves always choosing the action expected to lead to the highest reward based on the current knowledge of the environment.
- Only involves exploitation.
- Does not include any exploration.
- Can be disadvantageous in uncertain environments or those with unknown optimal actions.

## Off-policy vs On-policy Algorithms
- **Off-policy algorithms:** Use a different policy during training and inference.
- **On-policy algorithms:** Use the same policy for both training and inference.

## Monte Carlo and Temporal Difference Learning Strategies

### Monte Carlo (MC)
Learning at the end of the episode. The value function (or policy function) is updated after the complete episode.

### Temporal Difference (TD)
Learning at each step. The value function (or policy function) is updated at each step without requiring a complete episode.


## Additional Readings

Monte Carlo and TD Learning
To dive deeper into Monte Carlo and Temporal Difference Learning:

[Why do temporal difference (TD) methods have lower variance than Monte Carlo methods?](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met)

[When are Monte Carlo methods preferred over temporal difference ones?](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)

Q-Learning

Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 5, 6 and 7

[Foundations of Deep RL Series, L2 Deep Q-Learning by Pieter Abbeel](https://www.youtube.com/watch?v=Psrhxy88zww&ab_channel=PieterAbbeel)