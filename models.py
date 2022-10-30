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

        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out



class CNNModel(nn.Module):
    def __init__(self, output):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=8),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=4),
            # nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2498560, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, output),
            nn.Softmax()
        )
        
    def forward(self, x):
        x = self.features(x)
        h = torch.max(x, 1)
        print(h.shape)
        h = torch.flatten(h)
        print(h.shape)
        x = self.classifier(h)
        return x, h