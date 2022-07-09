import torch
from abc import ABC, abstractmethod
from metrics import ECELoss
from net import CondVAE
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def test(model, loader, device):
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for image, label in loader:
            preds.append(model(image.to(device)).cpu())
            labels.append(label.cpu())
    preds = torch.cat(preds, dim=0)  
    labels = torch.cat(labels) 
    acc = torch.argmax(preds, dim=-1).eq(labels).float().mean()
    ece_loss = ECELoss()
    ece = ece_loss(preds, labels)
    return acc.item(), ece.item()

class BaseTemp(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, *args, **kwargs):
        logits = self.classifier(*args, **kwargs)
        return logits / self.get_temp(*args, **kwargs)

    @abstractmethod
    def get_temp(self, input):
        pass

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        pass


class TemperatureScaling(BaseTemp):
    def __init__(self, classifier, optim="ECE") -> None:
        super().__init__()
        self.classifier = classifier
        assert optim == "ECE" or optim == "NLL", \
            "objective for tempreture scaling must be ECE or NLL"
        self.optim = optim
        self.temperature = 1.0

    def get_temp(self, *args, **kwargs):
        return self.temperature

    def calibrate(self, valid_loader, device="cpu", *args, **kwargs):
        self.classifier.eval()
        nll_criterion = torch.nn.CrossEntropyLoss()#.cuda()
        ece_criterion = ECELoss()#.cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader: 
                logits = self.classifier(input.to(device)).cpu()
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list)#.cuda()
            labels = torch.cat(labels_list)#.cuda()

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            #self.cuda()
            after_temperature_nll = nll_criterion(logits / T, labels).item()
            after_temperature_ece = ece_criterion(logits / T, labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if self.optim == 'ECE':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll

        return self


class AdaptiveTemperatureScaling(BaseTemp):
    def __init__(self, classifier, vae_params, classifier_last_layer_name, device) -> None:
        super().__init__()
        self.classifier = classifier
        return_nodes = {
                classifier_last_layer_name: 'feats',
                'fc': 'preds'
            }
        self.feature_extractor = create_feature_extractor(self.classifier, return_nodes=return_nodes)
        self.vae = CondVAE(**vae_params, device=device)

    def get_temp(self, input, *args, **kwargs):
        return self.vae.sample_t(self.feature_extractor(input)['feats'])

    def calibrate(self, valid_loader, device="cpu", *args, **kwargs):
        # agg features

        features = []
        labels = []
        logits = []
        with torch.no_grad():
            for input, label in valid_loader: 
                outputs = self.feature_extractor(input.to(device))
                features.append(outputs['feats'].cpu())
                logits.append(outputs['preds'].cpu())
                labels.append(label)

        val_dataset = TensorDataset(torch.cat(features, dim=0),
                                    torch.cat(logits, dim=0),
                                    torch.cat(labels))
        feat_loader = DataLoader(val_dataset, batch_size=128, drop_last=False, shuffle=True)

        # now train the vae
        optim = torch.optim.Adam(self.vae.parameters(), lr=5e-4)
        print("Training AdaTS")
        for epoch in tqdm(range(100)):
            loss = 0.0
            for feat, logits, labels in feat_loader:
                feat = feat.to(device)
                logits = logits.to(device)
                labels = labels.to(device)
                this_loss = self.vae.elbo(feat, labels)
                this_loss += self.vae.t_ce(feat, logits, labels)
                loss += this_loss
                this_loss.backward()
                optim.step()
                optim.zero_grad()




        
        
