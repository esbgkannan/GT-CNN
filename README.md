# GT-CNN

## Requirement

Please ensure the following software is installed:

- `Python (v3.7.4 or later)` [link](https://www.python.org/downloads/)
- `Pytorch (v1.8.1 or later recommend using GPU version)` [link](https://pytorch.org/)
- `fastai` [link](https://fastai1.fast.ai/install.html)
- `pandas` [link](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- `seaborn` [link](https://seaborn.pydata.org/installing.html)
- `torchviz` [link](https://pypi.org/project/torchviz/)
- `sklearn` [link](https://scikit-learn.org/stable/install.html)
- `scipy` [link](https://www.scipy.org/install.html)
- `numpy` [link](https://numpy.org/install/)
- `matplotlib` [link](https://matplotlib.org/stable/users/installing.html)
- `umap` [link](https://umap-learn.readthedocs.io/en/latest/)
- `jupyterlab` [link](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

## Step1: Sequence collection
Collect related sequences for any family, group you want to analyze. Ideally, you want >100 and <500 sequences purged with a sequnece identity of 10-95%.
Edit sequence IDs in format (>Family(GT2-A)|UniqueID|TaxInfo)

## Step2: Secondary structure prediction
- Done using NetsurfP2.0[link](http://www.cbs.dtu.dk/services/NetSurfP/). Generates a csv file with the predictions for all sequences. 
- Note: any other SOTA SS predictor also works as long as the output csv file is formated with such columns: ["Name", "fold", "family", "q3seq", "rawseq"]


## Step3: Preprocessing
- Notebook: Preprocessing_pipline.ipynb
- This notebook is mainly for 1. Domain and sequence lenghth based filtering, in our work, based on statistical analyasis, we select 798 as our cuttung threhold and padding length 2. Sequence paddings, this is mainly for CNN model to process 3. Data balancing, by augmenting underrepresented families with random mutation, each families are balanced to around 2k samples.


## Step4: Generate all GT level Reconstruction Error
- Notebook: RE_generator.ipynb
- Pretrained: ./PretrainedModels/Autoencoder_gtAll.pickle
- Better with GPU. 

## Step5: Generate GT cluster level Weighted Overlap Score
- Notebook: RE_generator_sub.ipynb
- Pretrained: ./PretrainedModels/Autoencoder_gt--.pickle
- Better with GPU.

## Step6: Analysis and fold prediction using the above results
- Notebook: RE_analysis.ipynb

## Step7: Sequence, Sec. str Alignment (CAM for seq used in training if needed)
- Notebook: CAM_analysis.ipynb
