import torch
from abc import ABC, abstractmethod
from metrics import ECELoss

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

    def calibrate(self, valid_loader, *args, **kwargs):
        self.classifier.eval()
        nll_criterion = torch.nn.CrossEntropyLoss()#.cuda()
        ece_criterion = ECELoss()#.cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                #input = input.cuda()
                logits = self.classifier(input)
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
        #self.cuda()
        return self

class AdaptiveTemperatureScaling(BaseTemp):
    def __init__(self, classifier) -> None:
        super().__init__()
        self.classifier = classifier
        self.T = 1.0

    def get_temp(self, *args, **kwargs):
        return self.T

    def calibrate(self, cal_ds, *args, **kwargs):
        self.T = 2.0



    
        



