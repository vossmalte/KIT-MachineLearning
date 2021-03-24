---
title: "Maschinelles Lernen - Grundlagen"
subtitle: "Self-test questions"
author: "Malle \\& Valle"
date: "\\today"
---

## Linear Regression

## Linear classification

## Model selection

Why is it a bad idea to evaluate your algorithm on the training set
    - overfitting
What is the difference between true and empirical risk
    - true risk: "real error" -> not evaluatable
    - empirical risk: error in training data -> can evaluate
The true risk can be decomposed in which parts,
How is the bias and the variance of a learning algorithm defined and how do the contribute to the
true risk?
    - bias^2: due to restriction of model class ("approximation error", "structure error")
    - variance: due to randomness of dataset ("estimation error")
    - noise: nothing we can do about
What is the advantage/disadvantage of k fold CV vs. the Hold out method?
    - Hold out: simple, BUT can have bad split 
    - K-fold: stable, BUT costly
Why does it make sense to penalize the norm of the weight vector?
    - prefer simple explanations (Occams razor)
    - prevent overfitting
Which norms can we use and what are the different effects?
    - L2: "circle" (equiv. to early stopping and random noise in some cases)
    - L1: sparse solutions
What is the effect of early stopping?
    - same as L2 reg., model is not "pushed to the limit"

## Trees and forests

What we mean with non parametric / instance based machine learning algorithms ?
    - algorithms without parameters -> no training
    - model needs to store samples
How k NN works ?
    - find k nearest samples 
        - reg: compute average
        - class: majority voting 
How to choose the k?
    - do cross validation, k with lowest error
Why is it hard to use for high D data ?
    - **curse of dimensionality** high D => points are far away from each other! => big neighborhood (need much data to "fill space")
How go search for nearest neighbors efficiently ?
    - KD-trees: split along a dimension at the median recursively (until limited samples per leaf)
    - search: find leaf of sample, go up in tree until K neighbors found. 
What a binary regression / decision tree is and 
What are useful splitting criterions
    - build recursively until (max depth / samples per leaf)
        - reg: minimize variance (#left samples * variance left + ...right)
        - class: minimize entropy
    - node: splitting value of some variable in < and >
How can we influence the model complexity of the tree?
    - set max depth / samples per leaf
Why is it useful to use multiple trees and randomization?
    - decrease variance to...
        - improve accuracy
        - avoid instability

## Dimensionality Reduction

What does dimensionality reduction mean?
    - remove redundant dimensions
How does linear dimensionality reduction work?
    - remove dimensions with lowest variance ;)
What is PCA? What are the three things that it does?
    - Principle Component Analysis:
        - zero-mean data
        - calculate Covariance Matrix Sigma
        - choose Eigenvectors with biggest Eigenvalues, project them on axes of low-D. space
What are the roles of the Eigenvectors and Eigenvalues in PCA?
    - Eigenvalue: is variance along Eigenvector
    - choose Eigenvectors with biggest Eigenvalues, project them on axes of low-D. space
Can you describe applications of PCA?
    - simplify data
    - pre-processing
    - analyze data: find "capture the essence"

## Clustering

How is the clustering problem defined? Why is it called “unsupervised”?
    - categorize data points into clusters
    - no labels / "ground truth" / right clustering
How do hierarchical clustering methods work? 
    - top-down: (not covered)
    - bottom up: merge "closest" clusters
        - distance metric for clusters needed.
What is the rule of the cluster 2 cluster distance and which distances can we use?
    - average linkage: average of all point-to-point distances
    - single linkage: minimal distance
    - complete linkage: maximal distance
    - centroid linkage: distance of centroids
How does the k means algorithm work? What are the 2 main steps?
    - assignment: assign samples to centers
    - adjustment: put centers on mean of "their" samples
Why does the algorithm converge? What is it minimizing?
    - SSD (sum of squared distances) between samples and its center is minimized
Does k means finds a the global minimum of the objective?
    - no only local minimum

## Density Estimation and EM

What are parametric methods and how to obtain their parameters?
    - gaussian distribution
    - MLE / gradinent descent
How many parameters have non parametric methods?
    - none :D (they have HYPER parameters)
What are mixture models?
    - GMM: weighted sum of gaussians
Should gradient methods be used for training mixture models?
    - no: the log does not go well with the sum / cyclic dependencies in parameters
How does the EM algorithm work?
    - E-step: calculate latent variables according to old parameters (find q(z) that minimizes KL)
    - M-step: new parameters ("argmax") ~ MLE
What is the biggest problem of mixture models?
    - how many mixture components do we need?
How does EM decomposes the marginal likelihood?
    - = Lower bound + KL divergence
Why does EM always improve the lower bound?
    - E: min. KL-d. , Likelihood constant => lower bound increases
    - M: Likelihood increases, KL-d. constant => lower bound increases
Why does EM always improve the marginal likelihood
    - lower bound is improved in every step => lower bound is lower bound for loglikelihood
    - E: likelihood constant
    - M: Likelihood increases (maximize)
Why can we optimize each mixture component independently with EM
!    - because its parameters are directly computed the weighted data points
Why do we need sampling for continuous latent variables?
    - because we typically can't compute the E-step with integrals instead of sums

## Kernel Methods

What are kernels and how are they useful?

What do we mean by the “Kernel trick”?

How to use kernels in regression (using Kernel Regression)?

How to use kernels in classification (using SVMs)?

Understand how to obtain dual optimization problems from the primal

... and its relation to kernel methods

What is the definition of a kernel and its relation to an underlying feature space?

Why are kernels more powerful than traditional feature based methods?

What do we mean by the kernel trick?

How do we apply the kernel trick to ridge regression?

## SVM

Why is it good to use a maximum margin objective for classification?

How can we define the margin as optimization problem?

What are slack variables and how can they be used to get a “soft” margin?

How is the hinge loss defined?

What is the relation between the slack variables and the hinge loss?

What are the advantages and disadvantages in comparison to logistic regression?

What is the difference between gradients and sub gradients

## Bayesian ML

What are the 2 basic steps behind Bayesian Learning?

Why is Bayesian Learning more robust against overfitting?

What happens with the posterior if we add more data to the training set?

What is completing the square and how does it work?

For which 2 cases can Bayesian Learning be solved in closed form?

Which approximations can we use if no closed form is available?

How can we derive Bayesian Linear regression

What is the advantage of Bayesian Linear regression to Ridge regression? What is
the conceptual difference?

What is the major advantage of GPs over Kernel Ridge Regression?

Why are GPs a Bayesian approach?

What principle allowed deriving GPs from a Bayesian regression point of view?

## Neural Nets

How does logistic regression relate to neural networks?

What kind of functions can single layer neural networks learn?

Why do we need non linear activation functions?

What activation functions can we use and what are the advantages/disadvantages of
those?

What output layer and loss function to use given the task (regression,

classification)?

Why not use a sigmoid activation function?

Derive the equations for forward and backpropagation for a simple network

What is mini batch gradient descent? Why use it instead of SGD or full gradient descent?

Why neural networks can overfit and what are the options to prevent it?

Why is the initialization of the network important?

What can you read from the loss curves during training (validation and training loss)?

How can we accelerate gradient descent? How does Adam work?

## RNN, DNN

Why are fully connected networks for images a bad idea and why do we need
images?

What are the key components of a CNN?

What hyper parameters can we set for a convolutional layer and what is their
meaning?

What hyper parameters can we set for a pooling layer and what is their meaning?

How can we compute the dimensionality of the output of a convolutional layer

Describe basic properties of AlexNet and VCG

What is the main idea of ResNet to make it very deep?
