# Learning algorithm

## Multi Agent Deep Deterministic Policy Gradient(MADDPG)

* DDPG is a different kind of actor-critic method. See page below for a summary of the DDPG algorithm
  - https://github.com/Paseul/Udacity_DRL_P2_continuous_control/edit/master/Report.md
* In MADDPG, the DQN uses the Q-network as the critical. Actor uses RL's policy gradient to learn method. 
It has individual actor network of hostile (hostile) or cooperative (cooperative) agents, but uses concentrated 
critics to guide direction/ goal.

![Multi Agent Actor Critic for Mixed Cooperative Competitive environments](https://user-images.githubusercontent.com/47571946/59245630-84859800-8c54-11e9-854d-5b16150434d1.png)

### Code implementation
1. Actor Network: Use Adam as Optimizer. Learing rate is 0.001.
  ```python
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
```

2. Critic Network: Use Adam as Optimizer. Learing rate is 0.01. No weight decay
  ```python
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

3. Idea for improving learning speed
When I first carried out the study, the learning speed was very slow. To improve this, I thought that I could learn 
faster by copying the work and storing it in memory. In the experimental environment, a reward of +0.1 for turning 
the ball over the net and a reward of -0.01 is obtained for falling. Therefore, if you get a reward of +0.1 and then 
drop the ball over again, the final reward will be 0.09. So, once 0.09-0.1 compensation is received, twice for 0.19-0.2; 
three for 0.29-0.3; four for 0.39-0.4; and five more times for 0.49 or higher.

  ```python
if reward >= 0.09 and reward <= 0.1:
    self.memory.add(state, action, reward, next_state, done)
if reward >= 0.19 and reward <= 0.2:
    for _ in range(2):
        self.memory.add(state, action, reward, next_state, done)
if reward >= 0.29 and reward <= 0.3:
    for _ in range(3):
        self.memory.add(state, action, reward, next_state, done)
if reward >= 0.39 and reward <= 0.4:
    for _ in range(4):
        self.memory.add(state, action, reward, next_state, done)
if reward >= 0.49:
    for _ in range(5):
        self.memory.add(state, action, reward, next_state, done)
```
Through this change, we found that the average score stayed at 0.18 up to 3000 episode when using the same Hyperparameter was improved

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
|BUFFER_SIZE | int(1e5)| replay buffer size
|BATCH_SIZE  | 128     | minibatch size
|GAMMA       |0.99     | discount factor
|TAU         |1e-3     | for soft update of target parameters
|LR_ACTOR    |1e-3     | learning rate of the actor 
|LR_CRITIC   |1e-3     | learning rate of the critic
|WEIGHT_DECAY|0        | L2 weight decay

### 2. Plot of rewards per episode

![Reselt](https://user-images.githubusercontent.com/47571946/59245562-5bfd9e00-8c54-11e9-9287-31f8074cca82.png)
Environment solved in 1853 episodes

# Ideas for future work

### Change the return value of the lab environment
In the original ML-Agent program, if the net is not turned over, the agent's energy is reduced to prevent repeated 
compensation in the same place. In the program used here, there is no energy concept of agent, but the return received 
on the floor is -0.01 compared to -0.1 in ML-Agent.I think that the size of the negative return an agent receives when 
it drops is small, so it drops the ball more frequently in the early stages and this slows down the learning process.
I want to check the change in the speed of learning when I change the return of the ball to -0.1.
