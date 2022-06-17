import torch
from torch import nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader 

from calibration import TemperatureScaling
from metrics import ECELoss
from net import Net

net = Net() 

# TODO convert net to ResNet50
# TODO add ts val
# TODO add adat predictions

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
val_size = 5000

test_ds, val_ds = torch.utils.data.random_split(set, [len(set) - val_size, val_size])

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=50,
                                        shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=50,
                                          shuffle=True, num_workers=0)

train_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=True, num_workers=0)                                        

def test(model, loader):
    preds = []
    labels = []
    for image, label in loader:
        preds.append(model(image))
        labels.append(label)
    preds = torch.cat(preds, dim=0)  
    labels = torch.cat(labels) 
    acc = torch.argmax(preds, dim=-1).eq(labels).float().mean()
    ece_loss = ECELoss()
    ece = ece_loss(preds, labels)
    
    return acc.item(), ece.item()

# Train a simple network
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

print("#" * 100)
print("Plain: Acc %.3f, ECE %.3f" % test(net, test_loader))

ts = TemperatureScaling(net)
ts.calibrate(val_loader)
print("TS: Acc %.3f, ECE %.3f" % test(ts, test_loader))

adats = TemperatureScaling(net)
adats.calibrate(val_loader)
print("AdatS: Acc %.3f, ECE %.3f" % test(adats, test_loader))

print("#" * 100)






