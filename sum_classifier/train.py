import torch.nn.functional as F
import torch


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)

        # Unbalanced dataset !!
        weight = torch.tensor([100/min(k+1, 19-k) for k in range(19)])
        loss = F.nll_loss(output, target, weight=weight)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
