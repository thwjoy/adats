# Adaptive Temperature Scaling

This is work in progress, but please find MVP code here.


## Instructions

You need to find the name of the last layer in your model, we use https://pytorch.org/vision/stable/feature_extraction.html in the codebase.

```
my_classifier = MyClassifier(num_classes=10)

vae_params = {
    "z_dim": 16, 
    "in_dim": 2048, # feature size, needs to match last layer of my_classifier
    "num_classes": 10
}

adats_classifier = AdaptiveTemperatureScaling(classifier=my_classifier, 
                                   vae_params=vae_params,
                                   classifier_last_layer_name='view', # this is the layer name of the last layer
                                   device=device)

adats_classifier.calibrate(val_loader, device)

output = adats_classifier(input)
```
