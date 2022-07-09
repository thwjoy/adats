'''
Pytorch implementation of ResNet models.
Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
'''
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import os

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out

BASE_URL = "https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR10/"

def resnet50(temp=1.0, download=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    if download:
        # download weights
        di = torch.hub.load_state_dict_from_url(
                os.path.join('https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR10/',
                             'resnet50_cross_entropy_350.model'),
                progress=True,
                map_location=torch.device('cpu'))
        # remove module from keys
        new_di = OrderedDict((keys[len("module."):], v) for keys, v in di.items())
        model.load_state_dict(new_di)
    return model

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], in_dim=28**2):
        super().__init__()
        self.im_dim = in_dim
        layers = []
        indim = in_dim
        for dim in hidden_dim[:-1]:
            layers.append(nn.Linear(indim, dim))
            layers.append(nn.ELU())
            indim = dim
        layers.append(nn.Linear(indim, hidden_dim[-1]))
        self.fc1 = nn.Sequential(*layers)
        self.fc21 = nn.Linear(hidden_dim[-1], z_dim)
        self.fc22 = nn.Linear(hidden_dim[-1], z_dim)    

    def forward(self, x):
        hidden = F.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.clamp(F.softplus(self.fc22(hidden)), min=1e-3)
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], in_dim=28**2, n_nets=1):
        super().__init__()
        # setup the two linear transformations used
        layers = []
        self.n_nets = n_nets
        indim = z_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(indim, dim))
            layers.append(nn.ELU())
            indim = dim
        layers.append(nn.Linear(indim, in_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=[512, 512]):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        layers = []
        indim = in_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(indim, dim))
            layers.append(nn.ELU())            
            indim = dim
        layers.append(nn.Linear(indim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CondPrior(nn.Module):
    def __init__(self, z_dim, n_classes) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.loc = nn.Linear(n_classes, z_dim, bias=False)
        self.scale = nn.Linear(n_classes, z_dim, bias=False)

    def forward(self, y):
        one_hot = F.one_hot(y, num_classes=self.n_classes).float()
        loc = self.loc(one_hot)
        scale = F.softplus(self.scale(one_hot)).clamp(min=1e-3)
        return loc, scale

def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    kl = 0.5 * (2 * scale_p.log() - 2 * scale_q.log() + \
                (locs_q - locs_p).pow(2) / scale_p.pow(2) + \
                scale_q.pow(2) / scale_p.pow(2) - torch.ones_like(locs_q)).sum(dim=-1)

    return kl

def log_likelihood(recon, xs):
        return dist.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=-1)

class CondVAE(torch.nn.Module):
    def __init__(self, z_dim, num_classes, device,
                 in_dim, n_nets=1):
        super(CondVAE, self).__init__()
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.device = device
        self.num_classes = num_classes
        self.n_nets = n_nets
        self.encoder = Encoder(self.z_dim, hidden_dim=[1024, 512, 512], in_dim=self.in_dim)
        self.decoder = Decoder(self.z_dim, hidden_dim=[1024, 512, 512], in_dim=self.in_dim)
        self.t_pred = Classifier(self.num_classes, 1, hidden_dim=[128, 128]) 
        self.cond_prior = CondPrior(self.z_dim, self.num_classes)
        self.to(device)

    def elbo(self, x, y):
        bs = x.shape[0]
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        kl = compute_kl(*post_params, *self.cond_prior(y))
        recon = self.decoder(z)
        log_pxz = log_likelihood(recon, x)
        loss = - (log_pxz - kl)
        return loss.mean()

    def t_ce(self, x, pred_logits, y):
        t = self.sample_t(x)
        return F.cross_entropy(pred_logits / t.view(-1, 1), y)

    def sample_t(self, x):
        y = torch.linspace(0, self.num_classes-1, self.num_classes).unsqueeze(0).long().to(x.device)
        y = y.expand(x.shape[0], -1)
        bs = x.shape[0]
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample().unsqueeze(1).expand(-1, self.num_classes, -1)
        p = dist.Normal(*self.cond_prior(y)).log_prob(z).sum(dim=-1)
        t_preds = self.t_pred(p)
        return F.softplus(t_preds).clamp(min=1e-3)
