import torch
import torch.nn.functional as F


def test(model, device, test_loader):
    weight = torch.tensor([100 / min(k + 1, 19 - k) for k in range(19)])
    weight = weight.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data1, data2, target) in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)

            test_loss += F.nll_loss(output, target, weight=weight, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
