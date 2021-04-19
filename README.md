# GT-CNN: Glycosyltransferase fold prediction using deep learning

## OS Requirements

The package has been tested on the following systems:

- OS: Ubuntu 18.04.5

## Dependencies

Please ensure the following software is installed:

- [`Python`](https://www.python.org/downloads/)
- [`Pytorch (with CUDA)`](https://pytorch.org/)
- [`fastai`](https://fastai1.fast.ai/install.html)
- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [`seaborn`](https://seaborn.pydata.org/installing.html)
- [`torchviz`](https://pypi.org/project/torchviz/)
- [`sklearn`](https://scikit-learn.org/stable/install.html)
- [`scipy`](https://www.scipy.org/install.html)
- [`numpy`](https://numpy.org/install/)
- [`matplotlib`](https://matplotlib.org/stable/users/installing.html)
- [`umap`](https://umap-learn.readthedocs.io/en/latest/)
- [`jupyterlab`](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [`opencv2`](https://pypi.org/project/opencv-python/)
- [`tensorboardX`](https://pypi.org/project/tensorboardX/)
- [`imblearn`](https://pypi.org/project/imblearn/)

The experiment workstation has Pytorch environment of version 1.8+ and cuda version of 11.1. Please install the version of Pytorch that applicable to your settings.

## Environments, Dataset and Pretrained Models Preparation

```bash
# clone this repository to replicate our experiments
git clone https://github.com/esbgkannan/GT-CNN
cd GT-CNN

# this will download required files into the following directories for the analysis
# (1) datasets, (2) pretrained models, (3) example outputs
bash download.sh

# using Anaconda to set environment
conda create --name env_name python=3.7 -y
conda activate env_name
# install packages (this should take a few minutes)
pip install -r requirements.txt
```

## Complete Pipeline:

Note: The model is trained and tested with two NVIDIA 2080Ti graph cards. If you don't have a gpu, calculating reconstruction errors can take approximately up to an hour. Please feel free to use precomputed RE values from the datasets folder to save this computational time instead.

### Step1: Sequence collection

Collect related sequences for any family or group you want to analyze. Sequences can ideally be purged with a sequence similarity of 60-95%.
Edit sequence IDs in format (>Family(GT2-A)|UniqueID|TaxInfo)

### Step2: Secondary structure prediction
- Done using NetsurfP2.0[link](http://www.cbs.dtu.dk/services/NetSurfP/). Generates a csv file with the predictions for all sequences. 
- Note: any other SOTA SS predictor also works as long as the output csv file is formated with such columns: ["Name", "fold", "family", "q3seq", "rawseq"] where q3seq column should have the string of predicted secondary structures with C: Coils, H: Helix and E: Beta sheets and the rawseq column should have the primary sequence of the exact length as the string in q3seq column.


### Step3: Preprocessing
- Notebook: [1-Preprocessing.ipynb](./Codes/1-Preprocessing.ipynb) 
- This notebook is mainly for: 
1. Domain and sequence length based filtering. For the GTs, based on statistical analysis of domain lengths, we select 798 as our threshold for maximum sequence and padding length. Any sequences longer than this should be trimmed to less than or equal to 798 amino acids. This can be done to include only the GT domain and remove other confounding domains with elaborate secondary structures.
2. Adding the sequence paddings to make all sequences of 798 aa length, which is mainly for the CNN model to process the input as a single matrix.


### Step4: CNN-Attention model training (requires GPU)
- Notebook: [2-CNNAttention.ipynb](./Codes/2-CNNAttention.ipynb) 
- This notebook is for training the CNN-Attention model using outputs generated from preprocessing steps.
- The pretrained model is made available and the training datasets are available upon request.

### Step5: Autoencoder models training (requires GPU)
- Notebook: [3-CNNAutoencoder-all.ipynb](./Codes/3-CNNAutoencoder-all.ipynb) and [4-CNNAutoencoder-sub.ipynb](./Codes/4-CNNAutoencoder-sub.ipynb) 
- This notebook is for training the autoencoder model using locked features generated from the CNN model.
- The pretrained models are made available and related datasets are available upon request.

### Step6: Generate all GT level Reconstruction Error (recommend GPU)
- Notebook: [5-RE_FAS_calculations.ipynb](./Codes/5-RE_FAS_calculations.ipynb) 
- Pretrained: ./PretrainedModels/Autoencoder_gtAll.pickle

### Step7: Generate GT cluster level Fold Assignment Score (recommend GPU)
- Notebook: [5-RE_FAS_calculations.ipynb](./Codes/5-RE_FAS_calculations.ipynb) 
- Pretrained: ./PretrainedModels/Autoencoder_gt--.pickle

### Step8: Analysis and fold prediction using the above results (recommend GPU)
- Notebook: [5-RE_FAS_calculations.ipynb](./Codes/5-RE_FAS_calculations.ipynb) 
- Plotting of results.
- Generate RE and FAS values for new families.

## For fold prediction of a new family of sequences:

Required steps are Step 1-3 and 6-8.

The pretrained models from Step 4 and 5 are made available.

All the required files to run the notebooks for these required steps are provided in the directories PretrainedModels and Datasets which will be generated upon running the download.sh script.

All the required notebooks outlining the steps are provided in the [Codes](./Codes) directory.

Outputs for each step are written to the ExampleOutputs directory and can be changed within the jupyter notebooks.

## For questions, comments and requests, please contact the ESBG lab at UGA.

Natarajan Kannan: nkannan@uga.edu

Zhongliang Zhou: Zhongliang.Zhou@uga.edu

Rahil Taujale: rtaujale@uga.edu

## Updates

- v1.0.0
  - First packaged version with complete notebooks and datasets.

## Citation

If you find this tool helpful, please cite:

Rahil Taujale, Zhongliang Zhou, Wayland Yeung, Kelley W Moremen, Sheng Li and Natarajan Kannan.**Mapping the glycosyltransferase fold landscape using deep learning.** Preprint.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
