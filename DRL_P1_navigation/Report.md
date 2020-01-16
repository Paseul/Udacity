# The Learning Algorithm
## Deep Q-Learning(DQN)
### Q funtion has instability becauseof:
  1. the correlations present in the sequence of observations
  2. the fact that small updates to Q may significantly change the policy and therefore change the data distribution
  3. the correlations between the action-values (Q) and the target values r+γ max⁡(a')Q(s',a')
### instability addressed by two ideas
  1. Experience Replay
  - removing correlations in the observation sequence and smoothing over changes in the data distribution
  2. Fixed Q Target
  - only periodically updated, thereby reducing correlations with the target
  
  * Excerpted from the Human-level control through deep reinforcement learning(Nature)
  ![Algorithm](https://user-images.githubusercontent.com/47571946/58063483-6d094100-7bb9-11e9-8388-f4c23d74d72c.png)
  * screenshot is taken from the Deep Reinforcement Learning Nanodegree course
# Model architectures
## Project is consisyt of:
  1. model.py
  - Create the first hidden layer as the size received from fc1_units (here: 64) and the second hidden layer as the size received from fc2_units (here: 64).
  - Connect the first hidden layer at state_size and the second hidden layer to form a neural network to derive the action value.
  
  2. dqn_agent.py
  - __init__: initialize the Replay Buffer
  - step(self, state, action, reward, next_state, done): Save the steps performed by the agent, and update the current network light and target network weights in the local network every 4 steps.
  - act((self, state, eps=0.): returning the action to corrects the current policy
  
# Hpyerparameters
## I select hyperparameters by fixing others but changing one parameter. And then Choose best result Hpyerparameters
## Compere figures are below(in Plot of Rewords) 
### dqn_agnet.py Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

### Neural Networks used
Input(37, state) -> fc(64, relu) -> fc(64, relu) -> output(4, action)

# Plot of Rewords
## Neural Networks
![nural-network](https://user-images.githubusercontent.com/47571946/58063890-07b64f80-7bbb-11e9-89e2-b74748956240.png)
## Batchsize
![Batch_size](https://user-images.githubusercontent.com/47571946/58063902-0f75f400-7bbb-11e9-9449-9c546c88fb06.png)
## Buffersize
![Buffer_size](https://user-images.githubusercontent.com/47571946/58063909-14d33e80-7bbb-11e9-8ab6-1b39fe8b1533.png)
## Gamma
![Gamma](https://user-images.githubusercontent.com/47571946/58063914-1d2b7980-7bbb-11e9-9df9-ad1bfcd03915.png)

# Ideas for Future Work
## Larning from Pixels
  - environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view of the environment.
## Using Defferent Algorithm
  1. Proxiaml Policy Optimization(PPO, Used in ml-agents)
  - The PPO algorithm is a hardening learning algorithm that is used by Unity ml-agnets by default. Leveraging the Pyrotorch version of PPO algorithm will be a new challenge, and then comparing it with the learned algorithm of ml-agents.
  2. Use AWS, Azure, GCP 
  - Workspace in Udacity provides GPU mode based on AWS. However, uploading images to AWS, Azure, and GCP using a docker and writing a container from the server will be a common method in development environment. 
  
