import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DDQN, self).__init__()
                
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=11)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #self.conv2d_1 = [self.conv1, self.bnorm1, self.relu1, self.maxp1]

        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.fc1 = nn.Linear(2048, 256)
        self.fc1bnorm = nn.BatchNorm1d(64)
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.fc2bnorm = nn.BatchNorm1d(64)
        self.fc2relu = nn.ReLU()
        
        self.fc3 = nn.Linear(256, action_size)
        
        #Dualing Network
        self.fc3_adv = nn.Linear(256, action_size)
        self.fc3_v = nn.Linear(256, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
            
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        #state = F.relu(self.bn3(self.conv3(state)))
        #print(state.shape)
        state = state.view(state.size(0), -1)
        state = F.relu(self.fc1(state))
        #state = F.relu(self.fc2(state))
        #print(state.shape)
    
        '''state = self.conv1(state)
        state = self.conv1bnorm(state)
        state = self.conv1relu(state)
        state = self.conv1maxp(state)

        state = self.conv2(state)
        state = self.conv2bnorm(state)
        state = self.conv2relu(state)
        state = self.conv2maxp(state)

        #print(state.shape) #state is of shape Nx32x21x21
        state = state.reshape((-1,32*21*21)) #reshape the output of conv2 before feeding into fc1 layer

        state = self.fc1(state)
        state = self.fc1bnorm(state)
        state = self.fc1relu(state)

        state = self.fc2(state)
        state = self.fc2bnorm(state)
        state = self.fc2relu(state)'''

        #state = self.fc3(state)
        #Dualing Network
        adv = self.fc3_adv(state)
        val = self.fc3_v(state).expand(-1, adv.size(1))
        state = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return state
