import torch
import torch.nn.functional as F
import numpy as np


def test(model, device, test_loader):
    model.eval()

    sum_freq = np.array([0.0 for _ in range(9)] + [min(k + 1, 19 - k) / 200 for k in range(19)])
    diff_freq = np.array([(10 - abs(k)) / 200 for k in range(-9, 10)] + [0.0 for _ in range(9)])
    all_freq = sum_freq + diff_freq
    inverse_freq = 1 / all_freq
    weight = torch.tensor(inverse_freq, dtype=torch.float32)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data1, data2, op, target) in test_loader:
            data1, data2, op, target = data1.to(device), data2.to(device), op.to(device), target.to(device)
            output = model(data1, data2, op)
            target = target + 9
            test_loss += F.nll_loss(output, target, weight=weight, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
