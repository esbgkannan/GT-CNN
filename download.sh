#!/bin/bash

# dataset
wget https://www.dropbox.com/sh/uxbsymtziktlu0r/AACkVr6IhR87OCp8z_UsTchFa?dl=0 -O Datasets.zip
unzip Datasets.zip -d Datasets/
rm Datasets.zip

# pretrained model
wget https://www.dropbox.com/sh/44ftj7zfcpaaxce/AACwcJ_1mqwV3xqQwajNi89Ga?dl=0 -O PretrainedModels.zip
unzip PretrainedModels.zip -d PretrainedModels/
rm PretrainedModels.zip

# example outputs
wget https://www.dropbox.com/sh/5lr1ciigpigln72/AACpl447nLlrXqgKpyRuyddTa?dl=0 -O ExampleOutputs.zip
unzip ExampleOutputs.zip -d ExampleOutputs/
rm ExampleOutputs.zip
