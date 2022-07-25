from torchvision import datasets, transforms
import torch


def get_train_test_loader(train_kwargs, test_kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST(r'..\data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(r'..\data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader
