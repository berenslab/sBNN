# sparseBottleneck
A sparse bottleneck neural network to predict electrophysiological properties of neurons from their gene expression.

Code to reproduce results of: "Sparse Bottleneck Neural Networks for Exploratory Analysis and Visualization of Paired Datasets."

Requirements:
TensorFlow and Keras, specifically. We used version 1.13.1 for TensorFlow and 2.2.4 for Keras (https://keras.io/#installation).
Glmnet, a package to fit generalized linear models with penalties like ridge and lasso (https://github.com/bbalasub1/glmnet_python). These notebooks have not been tested with TensorFlow 2.

For fitting the linear models, it can take a couple of hours of runtime to perform cross validation (10 folds). For the bottleneck neural network framework, cross validation takes less time (~10s of minutes). If performed once, the data can be pickled, however, so that one does not need to rerun the models everytime for plotting. Check KerasSavedModels for pickled results. These can indeed directly be used in the notebooks for plotting and to reproduce the figures of the paper.

Note 1: Keras arrives at slightly different training results every time you rerun the same simulation. To reproduce exact results and figures like in the paper check out the folder KerasSavedModels.

Note 2: all the raw data can be found in the folder M1Data which are needed to run the two notebooks all the way, except for M1_combined_counts.tab which is too large for GitHub to handle. Please use the e-mailaddress below for now to ask to work with this final file as well. We are working on a way around this. Also check https://www.biorxiv.org/content/10.1101/2020.02.03.929158v1, which is the original paper (under review) working with this data.

Work conducted under supervision of Philipp Berens.
Contact yves.bernaerts@uni-tuebingen.de
