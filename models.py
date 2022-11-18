import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out


class CNNModel(nn.Module):
    def __init__(self, output):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=25, kernel_size=3),
            nn.BatchNorm1d(25),
            nn.ReLU(),

#             nn.Conv1d(in_channels=25, out_channels=25, kernel_size=1),
#             nn.BatchNorm1d(25),
#             nn.ReLU()

        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(7450, 512),
            nn.ReLU(),
            
            nn.Dropout(p=0.5),            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, output)
        )
        
    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x
