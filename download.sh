#!/bin/bash

# dataset
wget https://zenodo.org/records/15310844/files/Datasets.zip?download=1 -O Datasets.zip
unzip Datasets.zip -d Datasets/
rm Datasets.zip

# pretrained model
wget https://zenodo.org/records/15310844/files/PretrainedModels.zip?download=1 -O PretrainedModels.zip
unzip PretrainedModels.zip -d PretrainedModels/
rm PretrainedModels.zip

# example outputs
wget https://zenodo.org/records/15310844/files/ExampleOutputs.zip?download=1 -O ExampleOutputs.zip
unzip ExampleOutputs.zip -d ExampleOutputs/
rm ExampleOutputs.zip
