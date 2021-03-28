# Pipeline to prredict GT fold for any new group of sequences.

## Step1: Sequence collection
Collect related sequences for any family, group you want to analyze. Ideally, you want >100 and <500 sequences purged with a sequnece identity of 10-95%.
Edit sequence IDs in format (>Family(GT2-A)|UniqueID|TaxInfo)

## Step2: Secondary structure prediction
Done using NetsurfP2.0. Generates a csv file with the predictions for all sequences.

## Step3: Preprocessing
Notebook: Preprocessing_pipline.ipynb

## Step4: Generate all GT level Reconstruction Error
Notebook: RE_generator.ipynb
Requires GPU.

## Step5: Generate GT cluster level Weighted Overlap Score
Notebook: RE_generator_sub.ipynb
Requires GPU.

## Step6: Analysis and fold prediction using the above results
Notebook: RE_analysis.ipynb

## Step7: Sequence, Sec. str Alignment (CAM for seq used in training if needed)
Notebook: CAM_analysis.ipynb


## Dependencies
CNN.py
Utils.py
../Data folder
../models folder