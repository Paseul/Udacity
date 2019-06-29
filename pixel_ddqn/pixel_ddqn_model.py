import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DDQN, self).__init__()
                
        self.seed = torch.manual_seed(seed)
        '''shape = (1, 1, 12, 84, 84)
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=(1,3,3)) #nn.Conv3d(4, 8, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1,3,3))#nn.Conv3d(8, 16, kernel_size=1)#, output_padding = (1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=3)#nn.Conv3d(16, 8, kernel_size=1)#, output_padding = (1,1))
        self.bn3 = nn.BatchNorm3d(64)
       
        x = torch.rand(shape)
        print(self.conv1(x).shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print(self.conv2(x).shape)
        x = F.relu(self.bn2(self.conv2(x)))
        print(self.conv3(x).shape)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        conv_out_size = x.data.view(1, -1).size(1)        
        print(conv_out_size)
        fc = [conv_out_size, 256]
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], action_size)'''
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=11)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #self.conv2d_1 = [self.conv1, self.bnorm1, self.relu1, self.maxp1]

        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        '''self.conv3 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3relu = nn.ReLU()
        self.conv3maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))'''

        self.fc1 = nn.Linear(2048, 256)
        self.fc1bnorm = nn.BatchNorm1d(64)
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.fc2bnorm = nn.BatchNorm1d(64)
        self.fc2relu = nn.ReLU()
        
        self.fc3 = nn.Linear(256, action_size)

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

        state = self.fc3(state)

        return state
