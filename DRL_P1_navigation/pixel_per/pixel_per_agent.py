import numpy as np
import random
from collections import namedtuple, deque

from pixel_per_model import DDQN

import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

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
        self.td_error_memory = TDerrorMemory(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, i_episode):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
                
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE :
                if i_episode < 300:
                    experiences = self.memory.stack_sample()
                else:
                    indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
                    experiences = self.memory.index_sample(indexes)
                self.learn(experiences, GAMMA)

    def stack_state(self, state):
        if len(self.memory) >= 3:
            idx = len(self.memory)
            pre_exp = self.memory.memory[idx-1].state
            pre_pre_exp = self.memory.memory[idx-2].state
            pre_pre_pre_exp = self.memory.memory[idx-3].state

            stack_state = np.concatenate((pre_pre_pre_exp, pre_pre_exp, pre_exp, state), axis=1)
        else:
            stack_state = np.concatenate((state, state, state, state), axis=1)

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
        
    def calc_td_error(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """        
        
        experiences = self.memory.memory

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        stack_states = []    
        next_stack_states = []
        
        for idx in range(len(states)):
            exp = states[idx]
            exp = exp.reshape((-1,3,84,84))
            nexp = next_states[idx]
            nexp = nexp.reshape((-1,3,84,84))
            
            if idx >= 3:                
                pre_exp = states[idx-1]
                pre_exp = pre_exp.reshape((-1,3,84,84))
                pre_pre_exp = states[idx-2]
                pre_pre_exp = pre_pre_exp.reshape((-1,3,84,84))
                pre_pre_pre_exp = states[idx-3]   
                pre_pre_pre_exp = pre_pre_pre_exp.reshape((-1,3,84,84))
                pre_next = next_states[idx-1]
                pre_next = pre_next.reshape((-1,3,84,84))
                pre_pre_nexp = next_states[idx-2]
                pre_pre_nexp = pre_pre_nexp.reshape((-1,3,84,84))
                pre_pre_pre_nexp = next_states[idx-3]
                pre_pre_pre_nexp = pre_pre_pre_nexp.reshape((-1,3,84,84))
                              
                stack_state = torch.cat((pre_pre_pre_exp, pre_pre_exp, pre_exp, exp), dim=1)
                next_stack_state = torch.cat((pre_pre_pre_nexp, pre_pre_nexp, pre_next, nexp), dim=1)
            else:
                stack_state = torch.cat((exp, exp, exp, exp), dim=1)
                next_stack_state = torch.cat((nexp, nexp, nexp, nexp), dim=1)
            
            stack_states.append(stack_state)
            next_stack_states.append(next_stack_state)
        
        states = torch.from_numpy(np.vstack([s.cpu() for s in stack_states])).float().to(device)       
        next_states = torch.from_numpy(np.vstack([ns.cpu() for ns in next_stack_states])).float().to(device)     

        ## (1) Get the best action at next state using orininal Q network
        best_action_next = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        ## (2) calculate Q value using target network for next state at these actions, predicted at step 1      
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_action_next)
        
        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ############ td_error안에 리스트가 들어가는것 부터 해결해야함 TD 오차를 계산
        td_errors = (rewards.squeeze() + GAMMA * Q_targets_next.squeeze()) - Q_expected.squeeze().float()
        
        # TD 오차 메모리를 업데이트. Tensor를 detach() 메서드로 꺼내와서 NumPy 변수로 변환하고 다시 파이썬 리스트로 변환
        self.td_error_memory.memory = td_errors.cpu().detach().numpy().tolist()


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
            
    def memorize_td_error(self, td_error):  # PrioritizedExperienceReplay에서 추가
        '''TD 오차 메모리에 TD 오차를 저장'''
        self.td_error_memory.push(td_error)
        
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
        self.capacity = buffer_size
        #self.memory = []  # 실제 transition을 저장할 변수
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        #if len(self.memory) < self.capacity:
        #    self.memory.append(None)
        
        #self.memory[self.index] = self.experience(state, action, reward, next_state, done)        
        #self.index = (self.index + 1) % self.capacity
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self.memory

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
            stack_states.append(stack_state)
            actions.append(self.memory[idx].action)
            rewards.append(self.memory[idx].reward)
            next_stack_state = np.concatenate((pre_pre_exp, pre_exp, exp, next_exp), axis=1)
            next_stack_states.append(next_stack_state)
            dones.append(self.memory[idx].done)
        
        states = torch.from_numpy(np.vstack([s for s in stack_states])).float().to(device)     
        actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(device)
        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(device)
        next_states = torch.from_numpy(np.vstack([ns for ns in next_stack_states])).float().to(device)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def index_sample(self, indexes):
        """Randomly sample a batch of experiences from memory."""
        idx_states = []    
        actions = []
        rewards = []
        next_idx_states = []
        dones = []
        
        for idx in indexes:
            exp = self.memory[idx].state
            pre_exp = self.memory[idx-1].state
            pre_pre_exp = self.memory[idx-2].state
            pre_pre_pre_exp = self.memory[idx-3].state
            if idx+1 != len(self.memory):
                next_exp = self.memory[idx+1].state
            else:
                next_exp = self.memory[idx].state
            idx_state = np.concatenate((pre_pre_pre_exp, pre_pre_exp, pre_exp, exp), axis=1)          
            idx_states.append(idx_state)
            actions.append(self.memory[idx].action)
            rewards.append(self.memory[idx].reward)
            next_idx_state = np.concatenate((pre_pre_exp, pre_exp, exp, next_exp), axis=1)
            next_idx_states.append(next_idx_state)
            dones.append(self.memory[idx].done)

        states = torch.from_numpy(np.vstack([s for s in idx_states])).float().to(device)
        actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(device)
        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(device)
        next_states = torch.from_numpy(np.vstack([ns for ns in next_idx_states])).float().to(device)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
TD_ERROR_EPSILON = 0.0001  # 오차에 더해줄 바이어스

class TDerrorMemory:

    def __init__(self, BUFFER_SIZE):
        self.capacity = BUFFER_SIZE  # 메모리의 최대 저장 건수
        self.memory = []  # 실제 TD오차를 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수     

    def push(self, td_error):
        '''TD 오차를 메모리에 저장'''
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 메모리가 가득차지 않은 경우      
        
        if len(self.memory) -1 != self.index:
            self.index = len(self.memory) -1

        self.memory[self.index] =  td_error
        self.index = (self.index + 1) % self.capacity  # 다음 저장할 위치를 한 자리 뒤로 수정

    def __len__(self):
        '''len 함수로 현재 저장된 갯수를 반환'''
        return len(self.memory)

    def get_prioritized_indexes(self, BATCH_SIZE):
        '''TD 오차에 따른 확률로 인덱스를 추출'''

        # TD 오차의 합을 계산        
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 충분히 작은 값을 더해줌

        # BATCH_SIZE 개만큼 난수를 생성하고 오름차순으로 정렬
        rand_list = np.random.uniform(0, sum_absolute_td_error, BATCH_SIZE)
        rand_list = np.sort(rand_list)

        # 위에서 만든 난수로 인덱스를 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # TD_ERROR_EPSILON을 더한 영향으로 인덱스가 실제 갯수를 초과했을 경우를 위한 보정
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        '''TD 오차를 업데이트'''
        self.memory = updated_td_errors