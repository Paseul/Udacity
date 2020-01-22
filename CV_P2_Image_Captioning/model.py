import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)               
       
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             batch_first=True
                           )
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # outputs.shape[1]: 13, captions.shape[1]: 12, (outputs.shape[1]==captions.shape[1]) = False
        captions = captions[:, :-1]
        embeddings = self.embed(captions)        
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)        
        hiddens, _ = self.lstm(inputs)        
        outputs = self.fc(hiddens)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []   
        seq_len = 0
        
        while True:
            # embedded image to lstm
            output, states = self.lstm(inputs,states)
           
            # lstm output to linear function
            output = self.fc(output.squeeze(1))
            _, scores = torch.max(output, 1)
            
            # convert torch tensor to np.array
            outputs.append(scores.cpu().numpy()[0].item())
            
            # if score is <end>
            if (scores[0] == 1):
                break
            
            # sentence length limit
            if(seq_len == 20):
                break;
                
            seq_len += 1
            
            # output to embed function
            inputs = self.embed(scores)   
            inputs = inputs.unsqueeze(1)

        return outputs
    
    
    
    
    
    