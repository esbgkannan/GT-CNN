#!/bin/bash

# dataset
wget https://www.dropbox.com/sh/shgar3h0c6lyy3b/AAA16q78UmCX_qgp87RpzOcFa?dl=0 -O Datasets.zip
unzip ./Datasets.zip -d ../Datasets
rm Datasets.zip

# pretrained model
wget https://www.dropbox.com/sh/1ziq5qbg0ul8wb2/AAA98kokV0YJndSOd2kRmEKUa?dl=0 -O PretrainedModels.zip
unzip ./PretrainedModels.zip -d ../PretrainedModels
rm PretrainedModels.zip

# example outputs
wget https://www.dropbox.com/sh/blugiec012sqv0v/AABzS6Zjzq4ri8MhjhRIytcoa?dl=0 -O ExampleOutputs.zip
unzip ./ExampleOutputs.zip -d ../ExampleOutputs
rm ExampleOutputs.zip
