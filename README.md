# Sparse bottleneck neural networks for exploratory non-linear visualization of Patch-seq data
A sparse bottleneck neural network to predict electrophysiological properties of neurons from their gene expression.
Code corresponds to work found on [ArXiv](https://arxiv.org/abs/2006.10411).
![sBNN latent space visualisation](./figures/combined_figure.png)
<br><br>
## Requirements:

TensorFlow and Keras, specifically. We used version 1.13.1 for TensorFlow and 2.2.4 for Keras (https://keras.io/#installation).
Glmnet, a package to fit generalized linear models with penalties like ridge and lasso (https://github.com/bbalasub1/glmnet_python). These notebooks have not been tested with TensorFlow 2.

For the sparse bottleneck neural network framework and linear models, cross validation takes on average ~10 minutes. If performed once, the data can be pickled, however, so that one does not need to rerun the models everytime for plotting. Check `KerasSavedModels/scala_2020` and `KerasSavedModels/gouwens_2020` for pickled results. These can indeed directly be used in the notebooks for plotting and reproducing figures.


## Data
All the data can be found in the folders `data/scala_2020`, `data/gouwens_2020` and `data/stoeckius_2017` which are needed to run the notebooks all the way. The first dataset we call the M1 dataset in our paper and corresponds to the following [Nature](https://www.nature.com/articles/s41586-020-2907-3) study from previous work conducted in this lab and collaborative labs. The 2nd we call the V1 dataset which corresponds to work from the Allen Institute reported in [Cell](https://www.cell.com/cell/pdf/S0092-8674(20)31254-X.pdf) and the 3rd is called the CITE-seq dataset reported in [Nature Methods](https://www.nature.com/articles/nmeth.4380).

## Preprocessing
Preprocessing can be found in `preprocess.ipynb`. To run it for the M1 dataset, make sure you un-archive `data/scala_2020/final_patch_seq_allcells_exon_allreads.rar` and put `final_patch_seq_allcells_exon_allreads.txt` in that same folder. For the CITE-seq dataset, you need to get `GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv` from the [NCBI portal](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866) and put it in `data/stoeckius_2017`. You will also need to generate `stoeckius_2017.pickle` by running the preprocessing pipeline.

## Cross validation
Cross validation has been performed for both frameworks sparse reduced-rank regression and the sparse bottleneck neural network applied to the M1, V1 and CITE-seq dataset in `M1_cv.ipynb`, `V1_cv.ipynb` and `CITE-seq_cv.ipynb` respectively. Results are pickled in `KerasSavedModels` so it's possible to produce figures also without running the cross validation.

## Figures
With optimal parameters deduced from the cross validation, one can run models for the full datasets in `V1_full_dataset.ipynb`, `M1_full_dataset.ipynb` and `CITE-seq_full_dataset`. Trained neural network weights will be saved in `KerasSavedModels` and can then be used to produce final figures in `combined_figure.ipynb`.

## CITE-seq
We also show sBNN's efficacy beyond Patch-seq on another dataset from Stoeckius et al., 2017 in [Nature Methods](https://www.nature.com/articles/nmeth.4380). Analogously, cross-validation results can be found in `CITE-seq_cv.ipynb` and latent visualizations in `CITE-seq_full_dataset.ipynb`.
<br><br>
Work conducted under supervision of Dmitry Kobak and Philipp Berens.
<br><br>
Contact: yves.bernaerts@uni-tuebingen.de
<br><br>
Website: http://www.eye-tuebingen.de/berenslab/members/yves-bernaerts/
