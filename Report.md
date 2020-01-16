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
2. Soft updates: Every step, mix in 0.01% of regular network weight with target network weight

### Code implementation
1. Actor Network: Use Adam as Optimizer. Learing rate is 0.001.
  ```python
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
```

2. Critic Network: Use Adam as Optimizer. Learing rate is 0.01. No weight decay
  ```python
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```
  
## Model architectures

### Actor

  - Input(33:state, 4:action) -> fc(256, relu) -> fc(128, relu) -> output(tneh x)

### Critic 

  - Input(33:state, 4:action) -> fc(256, relu)-> cat(xs with action) -> fc(128, relu) -> output(x)

# Plot of rewards

### 1. Hyper Parameters

Because it takes a long time to train, it is hard to be sure that the learning is going well. 
When adjusting Hyperparameter, it is recommended that you take a short period of observation and then go back to work when you don't want to learn.

It is recommended that `LR_ACTOR, LR_CRITIC` change to a value between `1e-3 to 1e-4`, `BUFFER_SIZE` is between `1e5 to 1e6` and `BATCH_SIZE` is between `128 to 1024`.

|Name|Value|Discreption|
|----|-----|-----------|
|BUFFER_SIZE | int(1e6)| replay buffer size
|BATCH_SIZE  | 512     | minibatch size
|GAMMA       |0.99     | discount factor
|TAU         |1e-3     | for soft update of target parameters
|LR_ACTOR    |1e-4     | learning rate of the actor 
|LR_CRITIC   |1e-4     | learning rate of the critic
|WEIGHT_DECAY|0        | L2 weight decay

### 2. Plot of rewards per episode
![save](https://user-images.githubusercontent.com/47571946/58931072-31fe3480-8799-11e9-861d-2c837dc6e08c.png)
Environment solved in 247 episodes
# Ideas for future work

### 1. Using GPU
Intel 9700k, P2000 GPU, AMD 2700x, and GTX 1080ti were used as test environments. But the GPU was used less, and the experiment took a lot of time.
The same was true of the use of PPO algorithm in Unity, but reinforcement learning should basically continue to utilize past experiences.
It is expected that the learning rate will be improved if the experimental environment consisting of several single agents is created to increase the usage of GPUs.

### 2. Use different Algorithm
Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) algorithm as another method for adapting DDPG for continuous control.
