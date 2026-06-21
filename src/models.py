import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A standard Deep Multi-Layer Perceptron (MLP) used for MNIST.
    """
    def __init__(self, input_size=784, hidden_size=1000, num_classes=10, activation='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
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


class CNN(nn.Module):
    """
    A simple Convolutional Neural Network used for CIFAR-10.
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
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


class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and projects them into the embedding dimension.
    """
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2) 
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embed
        return x


class VisionTransformer(nn.Module):
    """
    The base Vision Transformer.
    Uses PyTorch's native TransformerEncoder for highly optimized multi-head attention.
    """
    def __init__(self, in_channels, img_size, patch_size, emb_size, num_layer, heads, num_classes, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead=heads, 
            dim_feedforward=emb_size * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)


class MNIST_ViT(VisionTransformer):
    """
    ViT configured for MNIST. 
    1 channel, 28x28 images. Patch size 7 perfectly divides 28 into a 4x4 grid.
    """
    def __init__(self):
        super().__init__(
            in_channels=1, img_size=28, patch_size=7, 
            emb_size=128, num_layer=6, heads=8, num_classes=10
        )

class CIFAR_ViT(VisionTransformer):
    """
    ViT configured for CIFAR-10. 
    3 channels, 32x32 images. Patch size 4 perfectly divides 32 into an 8x8 grid.
    """
    def __init__(self):
        super().__init__(
            in_channels=3, img_size=32, patch_size=4, 
            emb_size=256, num_layer=6, heads=8, num_classes=10
        )