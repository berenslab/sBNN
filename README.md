# sparseBottleneck
A sparse bottleneck neural network to predict electrophysiological properties of neurons from their gene expression.

Code to reproduce results of: TODO: MENTION PAPER HERE WHEN READY TO MAKE PUBLIC

Requirements:
TensorFlow and Keras, specifically. We used version 1.13.1 for TensorFlow and 2.2.4 for Keras (https://keras.io/#installation).
Glmnet, a package to fit generalized linear models with penalties like rigde and lasso (https://github.com/bbalasub1/glmnet_python).

Definitely for fitting the linear models, it can take a couple of hours of runtime to perform cross validation (10 folds). For the bottleneck neural network frameworks cross validation takes less time (~10s of minutes). If performed once, the data can be pickled however, so that one does not need to rerun the models everytime for plotting.

Note: Keras arrives at slightly different training results every time you rerun the same simulation. To reproduce exact results in the paper check out the folder KerasSavedModels.
