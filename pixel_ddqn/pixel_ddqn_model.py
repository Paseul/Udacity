import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DDQN, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)        

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        #linear_input_size = convw * convh * 32
        #self.head = nn.Linear(linear_input_size, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, action_size)
        
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(12, 16, kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv1bnorm = nn.BatchNorm2d(16)
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #self.conv2d_1 = [self.conv1, self.bnorm1, self.relu1, self.maxp1]

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv2bnorm = nn.BatchNorm2d(32)
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(32*21*21, 64)
        self.fc1bnorm = nn.BatchNorm1d(64)
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(64, 64)
        self.fc2bnorm = nn.BatchNorm1d(64)
        self.fc2relu = nn.ReLU()
        
        self.fc3 = nn.Linear(64, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #x = F.relu(self.head(x.view(x.size(0), -1)))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #return self.fc3(x)
        
        state = self.conv1(state)
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
        state = self.fc2relu(state)

        state = self.fc3(state)

        return state
        #return x
        #return self.head(x.view(x.size(0), -1))