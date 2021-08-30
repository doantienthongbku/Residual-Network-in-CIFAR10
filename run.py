import utils.early_stopping as early_stopping
import dataset.dataset as dataset
import model.model as model
import utils.trainer as trainer
from torch import optim
import torch
from torchsummary import summary
import torch.nn as nn
import matplotlib.pyplot as plt

# Model
in_channel = 3
num_classes = 10
n = 5

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device} device")

model1 = model.PlainNet(n).to(device)
model2 = model.ResNet_original(n).to(device)
model3 = model.ResNet_identity(n).to(device)

# Model invesgation
summary(model=model2, input_size=(3, 32, 32), device=device)


# Optimizer
optimizer = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)

# Criation
loss_fn = nn.CrossEntropyLoss()

# LrScheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,
                                                 eps=1e-8, verbose=True, cooldown=0, min_lr=0)

# Early Stopping
early_stopping = early_stopping.EarlyStopping(mode='loss', patience=7, verbose=True, delta=1e-3, path="weights/checkpoint.pth")

# Training
epochs = 10

train_loss_his, train_acc_his = [], []
test_loss_his, test_acc_his = [], []

best_valid_accuracy = 0.
best_model_state_dict = dict()
best_optim_state_dict = dict()

for epoch in range(epochs):
    train_loss, train_acc = trainer.train_epoch(dataset.train_loader, model2, criterion=loss_fn, optimizer=optimizer, device=device)
    test_loss, test_acc = trainer.valid_epoch(dataset.valid_loader, model2, criterion=loss_fn, device=device)

    print(f"Epoch {(epoch + 1):>2d}: - train_loss: {train_loss:>7f}, train accuracy: {(train_acc*100):>0.1f}%")
    print(f"Epoch {(epoch + 1):>2d}: -  test_loss: {test_loss:>7f},  test accuracy: {(test_acc*100):>0.1f}%\n")

    scheduler.step(test_loss)

    early_stopping(test_loss, model2)
    train_loss_his.append(train_loss)
    train_acc_his.append(train_acc)
    test_loss_his.append(test_loss)
    test_acc_his.append(test_acc)

# Visualize
plt.plot(train_loss_his, color='r', label='train_loss')
plt.plot(test_loss_his, color='b', label='test_loss')
plt.legend()
plt.show()

plt.plot(train_acc_his, color='r', label='train_acc')
plt.plot(test_acc_his, color='b', label='test_acc')
plt.legend()
plt.show()
