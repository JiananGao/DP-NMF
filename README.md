## Deep Probabilistic NMF

This repository contains an implementation of an extension of Non-negative matrix factorization algorithm by embedding denoising autoencoders in the NMF framework. We initially train the autoencoder network separately as a pretraining phase and then in the fine-tuning phase, we jointly optimize the weights of the network and the factor matrices to get the final set of clusters.

Requirements:
* Tensorflow
* Numpy

#### References
S.Bhattamishra. Deep probabilistic NMF using Denoising Autoencoders. International Journal of Machine Learning and  Computing, 8(1):To appear, 2018. [Paper](https://drive.google.com/file/d/1MOYiQQ5fzH8yhQdda-DrKTZkYFN2uNhw/view)