import torch.nn.functional as F
import torch
import numpy as np


def train(args, model, device, train_loader, optimizer, epoch):

    sum_freq = np.array([0.0 for _ in range(9)] + [min(k + 1, 19 - k) / 200 for k in range(19)])
    diff_freq = np.array([(10 - abs(k)) / 200 for k in range(-9, 10)] + [0.0 for _ in range(9)])
    all_freq = sum_freq + diff_freq
    inverse_freq = 1 / all_freq
    weight = torch.tensor(inverse_freq, dtype=torch.float32)

    model.train()
    for batch_idx, (data1, data2, op, target) in enumerate(train_loader):
        data1, data2, op, target = data1.to(device), data2.to(device), op.to(device), target.to(device)
        # Add 9 to all target
        target = target + 9
        optimizer.zero_grad()
        output = model(data1, data2, op)

        # Unbalanced dataset !!
        loss = F.nll_loss(output, target, weight=weight)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
