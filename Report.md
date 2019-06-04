# Learning algorithm

## Deep Deterministic Policy Gradient(DDPG)

* DDPG is a different kind of actor-critic method. Instead of an actual actor critic, it could be seen as approximate DQN 
  - Reason: the critic in DDPG is used to approximate the maximizer over the Q value of the next state and not as learned baseline

* DDPG use two deep neural networks: Actor, Critic
  - Actor: used to approximate the optimal policy determiinistically(output is best believed action for any given state)
  - Critic: learns to evaluate the optimal action value function by using the actors best believed action
  
** Limitation of DQN: not straightforward to use in continuous action spaces

### Two aspects of DDPG
1. Use replay buffer
2. Soft updates
  - Every step, mix in 0.01% of regular network weight with target network weight

### Code implementation
1. Actor Network
  ```python
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
```
  - Use Adam as Optimizer. Learing rate is 0.001.

2. Critic Network
  ```python
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```
  - Use Adam as Optimizer. Learing rate is 0.01. No weight decay
  
## Model architectures

### Actor

Input(33:state, 4:action) -> fc(400, relu) -> fc(300, relu) -> output(tneh x)

### Critic 

Input(33:state, 4:action) -> fc(400, relu)-> cat(xs with action) -> fc(300, relu) -> output(x)

# Plot of rewards

## Hyper Parameters

|Name|Value|Discreption|
|----|-----|-----------|
|BUFFER_SIZE | int(1e6)| replay buffer size
|BATCH_SIZE  | 1024    | minibatch size
|GAMMA       |0.99     | discount factor
|TAU         |1e-3     | for soft update of target parameters
|LR_ACTOR    |1e-4     | learning rate of the actor 
|LR_CRITIC   |1e-3     | learning rate of the critic
|WEIGHT_DECAY|0        | L2 weight decay

## Plot of rewards per episode

# Ideas for future work

## 1. Using GPU
Intel 9700k, P2000 GPU, AMD 2700x, and GTX 1080ti were used as test environments. But the GPU was used less, and the experiment took a lot of time.
The same was true of the use of PPO algorithm in Unity, but reinforcement learning should basically continue to utilize past experiences.
It is expected that the learning rate will be improved if the experimental environment consisting of several single agents is created to increase the usage of GPUs.

## 2. Use different Algorithm
Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm as another method for adapting DDPG for continuous control.
