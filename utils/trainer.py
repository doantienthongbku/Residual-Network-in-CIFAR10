import torch
import torch.nn as nn


def train_epoch(dataloader, model, criterion, optimizer, device='cpu'):
    size, num_batch = len(dataloader.dataset), len(dataloader)
    model = model.to(device)
    model.train()

    train_loss, train_acc = 0., 0.
    for batch, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)

        # forward
        pred = model(data)
        loss = criterion(pred, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer
        optimizer.step()

        # count loss and accuracy
        pred = nn.Softmax(dim=1)(pred)
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == targets).type(torch.float).sum().item()

    train_loss /= num_batch
    train_acc /= size

    return train_loss, train_acc


def valid_epoch(valid_loader, model, criterion, device='cpu'):
    size, num_batch = len(valid_loader.dataset), len(valid_loader)
    model = model.to(device)
    model.eval()

    test_loss, test_acc = 0., 0.
    with torch.no_grad():
        for batch, (data, targets) in enumerate(valid_loader):
            data, targets = data.to(device), targets.to(device)

            pred = model(data)
            loss = criterion(pred, targets)

            pred = nn.Softmax(dim=1)(pred)
            test_loss += loss.item()
            test_acc += (pred.argmax(1) == targets).type(torch.float).sum().item()

        test_loss /= num_batch
        test_acc /= size

    return test_loss, test_acc
