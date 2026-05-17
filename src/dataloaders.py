import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """
    Loads MNIST - Used in the original AdaSecant papers for MLP tests.
    Also excellent for the ModernLSTM (Sequential MNIST).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64, data_dir='./data'):
    """
    Loads CIFAR-10 - Used in the original papers for baseline CNN tests.
    Also used for the ModernResNet test.
    """
    # Note: We keep data augmentation standard. For purely testing *optimizers*,
    # sometimes it's preferred to look at raw optimization without heavy augmentation,
    # but basic flips/crops help the modern models generalize.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader


# ==============================================================================
# ADDITIONAL DATASETS FOR EXPANDED EXPERIMENTS
# ==============================================================================

def get_fashion_mnist_loaders(batch_size=64, data_dir='./data'):
    """
    Loads Fashion-MNIST - A drop-in, more challenging alternative to MNIST.
    Great for testing if AdaSecant handles slightly more complex patterns in MLPs.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader


def get_cifar100_loaders(batch_size=64, data_dir='./data'):
    """
    Loads CIFAR-100 - Crucial "stress test" dataset.
    100 classes mean the final classification layer is much larger, creating 
    a different gradient distribution and a more complex loss landscape.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader


def get_data_loaders(dataset_name, batch_size=64, data_dir='./data'):
    """
    Master helper function to cleanly request loaders by string name 
    inside your Jupyter Notebook loop.
    """
    name = dataset_name.lower().strip()
    if name == 'mnist':
        return get_mnist_loaders(batch_size, data_dir)
    elif name == 'cifar10':
        return get_cifar10_loaders(batch_size, data_dir)
    elif name == 'fashion_mnist':
        return get_fashion_mnist_loaders(batch_size, data_dir)
    elif name == 'cifar100':
        return get_cifar100_loaders(batch_size, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'mnist', 'cifar10', 'fashion_mnist', 'cifar100'.")