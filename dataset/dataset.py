import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

# Dataloader setting
train_batch_size = 64
train_shuffle = True

test_batch_size = 64
test_shuffle = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"use {device} device")

# Normalize data
mean = [0.49140018224716187, 0.4821578562259674, 0.44653069972991943]
std = [0.19525332748889923, 0.19247250258922577, 0.19420039653778076]

# Download dataset
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True)
    ]),
}

training_data = datasets.CIFAR10(
    root='dataset',
    train=True,
    transform=transform['train'],
    download=True
)

valid_data = datasets.CIFAR10(
    root='dataset',
    train=False,
    transform=transform['val'],
    download=True
)

# Dataloader
train_loader = DataLoader(dataset=training_data, batch_size=train_batch_size, shuffle=train_shuffle)
valid_loader = DataLoader(dataset=valid_data, batch_size=test_batch_size, shuffle=test_shuffle)
