import numpy as np
import random
from collections import namedtuple, deque

from pixel_ddqn_model import DDQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
REGULARIZATION = 1e-4   # regularization parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.last_frame = None
        self.last2_frame = None
        self.last3_frame = None
    
    def step(self, state, action, reward, next_state, done, i_episode):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
                       
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE :
                experiences = self.memory.stack_sample()
                self.learn(experiences, GAMMA)

    def stack_state(self, state):
        #state = state[0][0].reshape((-1,84,84))
        
        #stack_state = np.zeros((1, 3, 4, 84, 84))
        
        if len(self.memory) >= 3:
            idx = len(self.memory)
            pre_exp = self.memory.memory[idx-1].state
            #pre_exp = pre_exp[0][0].reshape((-1,84,84))
            pre_pre_exp = self.memory.memory[idx-2].state
            #pre_pre_exp = pre_pre_exp[0][0].reshape((-1,84,84))
            pre_pre_pre_exp = self.memory.memory[idx-3].state
            #pre_pre_pre_exp = pre_pre_pre_exp[0][0].reshape((-1,84,84))

            stack_state = np.concatenate((pre_pre_pre_exp, pre_pre_exp, pre_exp, state), axis=1)
            #stack_state = stack_state.reshape((-1,1,12,84,84))
            
            #stack_state[0, :, 0, :, :] = state
            #stack_state[0, :, 1, :, :] = pre_exp
            #stack_state[0, :, 2, :, :] = pre_pre_exp
            #stack_state[0, :, 3, :, :] = pre_pre_pre_exp
        else:
            stack_state = np.concatenate((state, state, state, state), axis=1)
            #stack_state = stack_state.reshape((-1,1,12,84,84))
            
            #stack_state[0, :, 0, :, :] = state
            #stack_state[0, :, 1, :, :] = state
            #stack_state[0, :, 2, :, :] = state
            #stack_state[0, :, 3, :, :] = state

        return stack_state
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        ## (1) Get the best action at next state using orininal Q network
        best_action_next = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        ## (2) calculate Q value using target network for next state at these actions, predicted at step 1      
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_action_next)
        
        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def stack_sample(self):
        """Randomly sample a batch of experiences from memory."""
        stack_states = []    
        actions = []
        rewards = []
        next_stack_states = []
        dones = []
        stack_state = np.zeros((1, 3, 4, 84, 84))
        next_stack_state = np.zeros((1, 3, 4, 84, 84))
        
        while len(stack_states) < self.batch_size:
            idx = random.sample(range(len(self.memory)), k=1)[0]
            exp = self.memory[idx].state
            if exp is None or (idx-2) < 0 or (idx+1) >= len(self.memory):
                continue
            else:
                pre_exp = self.memory[idx-1].state
                pre_pre_exp = self.memory[idx-2].state
                pre_pre_pre_exp = self.memory[idx-3].state
                next_exp = self.memory[idx+1].state
            
            stack_state = np.concatenate((pre_pre_pre_exp, pre_pre_exp, pre_exp, exp), axis=1)
            #stack_state = stack_state.reshape((-1,1,12,84,84))
            #stack_state[0, :, 0, :, :] = exp
            #stack_state[0, :, 1, :, :] = pre_exp
            #stack_state[0, :, 2, :, :] = pre_pre_exp
            #stack_state[0, :, 3, :, :] = pre_pre_pre_exp
            
            stack_states.append(stack_state)
            actions.append(self.memory[idx].action)
            rewards.append(self.memory[idx].reward)
            
            #next_stack_state[0, :, 0, :, :] = next_exp
            #next_stack_state[0, :, 1, :, :] = exp
            #next_stack_state[0, :, 2, :, :] = pre_exp
            #next_stack_state[0, :, 3, :, :] = pre_pre_exp
            next_stack_state = np.concatenate((pre_pre_exp, pre_exp, exp, next_exp), axis=1)
            #next_stack_state = next_stack_state.reshape((-1,1,12,84,84))
            next_stack_states.append(next_stack_state)
            dones.append(self.memory[idx].done)

        states = torch.from_numpy(np.vstack([s for s in stack_states])).float().to(device)
        actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(device)
        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(device)
        next_states = torch.from_numpy(np.vstack([ns for ns in next_stack_states])).float().to(device)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)