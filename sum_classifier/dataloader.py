from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt


class SumMNIST(Dataset):
    """SumMNIST dataset."""

    def __init__(self, mnist_dataset: Dataset, im_per_epoch: int = 60_000):
        """
        Args:
            mnist_dataset (Dataset): The MNIST dataset.
        """

        self.mnist_dataset = mnist_dataset
        self.im_per_epoch = im_per_epoch

        if im_per_epoch > len(mnist_dataset):
            replacement = True
        else:
            replacement = False

        self.sample_for_im1 = list(RandomSampler(mnist_dataset, replacement=replacement, num_samples=im_per_epoch))
        self.sample_for_im2 = list(RandomSampler(mnist_dataset, replacement=replacement, num_samples=im_per_epoch))

    def __len__(self) -> int:
        return self.im_per_epoch

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the two MNIST digits
        digit1, label1 = self.mnist_dataset[self.sample_for_im1[idx]]
        digit2, label2 = self.mnist_dataset[self.sample_for_im2[idx]]

        return digit1, digit2, label1 + label2


def get_train_test_loader(train_kwargs, test_kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST(r'..\data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(r'..\data', train=False, download=True, transform=transform)

    sum_dataset_train = SumMNIST(dataset1, im_per_epoch=60_000)
    sum_dataset_test = SumMNIST(dataset2, im_per_epoch=10_000)

    train_loader = DataLoader(sum_dataset_train, **train_kwargs)
    test_loader = DataLoader(sum_dataset_test, **test_kwargs)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_train_test_loader(train_kwargs={'batch_size': 64, 'shuffle': True},
                                                         test_kwargs={'batch_size': 64, 'shuffle': False})

    for im1, im2, target in train_loader:
        # Plot of images
        im1 = im1[0]
        im2 = im2[0]

        # Subplot both of images
        fig = plt.figure(figsize=(5, 5))
        fig.suptitle(f"Sum = {target[0].numpy()}", fontsize=14)
        plt.subplot(1, 2, 1)
        plt.imshow(im1.squeeze().numpy(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(im2.squeeze().numpy(), cmap='gray')
        plt.show()
        break
