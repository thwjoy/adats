import torch
from abc import ABC, abstractmethod


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
        self.T = 1.0

    def get_temp(self, *args, **kwargs):
        return self.T

    def calibrate(self, cal_ds, *args, **kwargs):
        self.T = 2.0

class AdaptiveTemperatureScaling(BaseTemp):
    def __init__(self, classifier) -> None:
        super().__init__()
        self.classifier = classifier
        self.T = 1.0

    def get_temp(self, *args, **kwargs):
        return self.T

    def calibrate(self, cal_ds, *args, **kwargs):
        self.T = 2.0



    
        



