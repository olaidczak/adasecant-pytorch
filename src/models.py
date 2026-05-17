import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SECTION 1: ORIGINAL PAPER ARCHITECTURES (Baseline Optimization Landscapes)
# These models lack modern tricks like Batch Normalization or Skip Connections.
# This results in a harder, rougher loss landscape—exactly what AdaSecant 
# was originally designed to navigate and solve.
# ==============================================================================

class PaperMLP(nn.Module):
    """
    A standard Deep Multi-Layer Perceptron (MLP) used for MNIST.
    In the AdaSecant papers, the authors heavily tested MLPs to demonstrate
    how the optimizer handles vanishing gradients in deep, simple networks.
    """
    def __init__(self, input_size=784, hidden_size=1000, num_classes=10, activation='relu'):
        super(PaperMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Bengio's lab often tested different activations to see how optimizers 
        # handled different curvature types (Tanh has different saturation than ReLU)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Activation must be 'relu' or 'tanh'")

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class PaperCNN(nn.Module):
    """
    A simple Convolutional Neural Network used for CIFAR-10.
    Notice the ABSENCE of Batch Normalization. In 2014, networks relied entirely
    on the optimizer to navigate the scale of gradients between layers. 
    """
    def __init__(self, num_classes=10):
        super(PaperCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 pools of 2x2, a 32x32 image becomes 4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# SECTION 2: ADDITIONAL MODERN ARCHITECTURES (The "Stress Test")
# These models introduce modern architectural components that fundamentally change
# the curvature and variance of the loss landscape.
# ==============================================================================

class BasicBlock(nn.Module):
    """Helper block for ModernResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModernResNet(nn.Module):
    """
    A lightweight ResNet (similar to ResNet-9) for CIFAR-10.
    Why add this? Skip connections (shortcuts) and Batch Normalization heavily 
    smooth out the loss landscape. It is a crucial experiment to see if AdaSecant's
    complex curvature calculations still provide a benefit here, or if the landscape 
    is now so smooth that Adam/SGD simply win out.
    """
    def __init__(self, num_classes=10):
        super(ModernResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, stride):
        layer = BasicBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ModernLSTM(nn.Module):
    """
    An LSTM model for sequence classification (e.g., IMDB text classification 
    or sequential MNIST). 
    Why add this? Recurrent Neural Networks natively suffer from exploding and 
    vanishing gradients. RMSprop and Adam were built to scale these magnitudes. 
    Testing AdaSecant's variance-reduction against RMSprop on an LSTM is the 
    ultimate test of gradient stability.
    """
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(ModernLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True means inputs are (batch_size, sequence_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out