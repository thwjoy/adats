import torch
from torch import nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader 
from calibration import TemperatureScaling, \
                        AdaptiveTemperatureScaling, \
                        test
from net import resnet50
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

net = resnet50(1.0, True, num_classes=10).to(device)

# TODO add adat predictions

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )])

    
set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)

val_size = 5000

val_ds, test_ds = torch.utils.data.random_split(set, [val_size, len(set) - val_size])

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256,
                                        shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256,
                                          shuffle=True, num_workers=4)
                                      
net.eval()
print("#" * 100)
print("Plain: Acc %.3f, ECE %.3f" % test(net, test_loader, device))

ts = TemperatureScaling(net)
ts.calibrate(val_loader, device)
print("TS: Acc %.3f, ECE %.3f" % test(ts, test_loader, device))

vae_params = {
    "z_dim": 16, 
    "in_dim": 2048, # feature size
    "num_classes": 10
}

adats = AdaptiveTemperatureScaling(classifier=net, 
                                   vae_params=vae_params,
                                   classifier_last_layer_name='view',
                                   device=device)

adats.calibrate(val_loader, device)
print("AdaTS: Acc %.3f, ECE %.3f" % test(adats, test_loader, device))

# train multiple models
vals = []
for i in range(10):
    adats = AdaptiveTemperatureScaling(classifier=net, 
                                   vae_params=vae_params,
                                   classifier_last_layer_name='view',
                                   device=device)

    adats.calibrate(val_loader, device)
    vals.append(test(adats, test_loader, device))

array = np.array(vals)
print("AdaTS: Acc %.3f , ECE %.3f +/- %.3f"  % (*array.mean(axis=0), array.std(axis=0)[1]))
print("#" * 100)






