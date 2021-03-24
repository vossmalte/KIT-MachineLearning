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

What is the difference between true and empirical risk

The true risk can be decomposed in which parts?

How is the bias and the variance of a learning algorithm defined and how do the contribute to the
true risk?

What is the advantage/disadvantage of k fold CV vs. the Hold out method?

Why does it make sense to penalize the norm of the weight vector?

Which norms can we use and what are the different effects?

What is the effect of early stopping?

## Trees and forests

What we mean with non parametric / instance based machine learning algorithms ?

How k NN works ?

How to choose the k?

Why is it hard to use for high D data ?

How do search for nearest neighbors efficiently ?

What a binary regression / decision tree is

What are useful splitting criterions

How can we influence the model complexity of the tree?

Why is it useful to use multiple trees and randomization?

## Dimensionality Reduction

What does dimensionality reduction mean?

How does linear dimensionality reduction work?

What is PCA? What are the three things that it does?

What are the roles of the Eigenvectors and Eigenvalues in PCA?

Can you describe applications of PCA?

## Clustering

How is the clustering problem defined? Why is it called “unsupervised”?

How do hierarchical clustering methods work? What is the rule of the cluster 2
cluster distance and which distances can we use?

How does the k mean algorithm work? What are the 2 main steps?

Why does the algorithm converge? What is it minimizing?

Does k means finds a the global minimum of the objective?

## Density Estimation and EM

What are parametric methods and how to obtain their parameters?

How many parameters have non parametric methods?

What are mixture models?

Should gradient methods be used for training mixture models?

How does the EM algorithm work?

What is the biggest problem of mixture models?

How does EM decomposes the marginal likelihood?

Why does EM always improve the lower bound?

Why does EM always improve the marginal likelihood

Why can we optimize each mixture component independently with EM

Why do we need sampling for continuous latent variables?

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

## Neural Nets 2

Why are fully connected networks for images a bad idea and why do we need
images?

What are the key components of a CNN?

What hyper parameters can we set for a convolutional layer and what is their
meaning?

What hyper parameters can we set for a pooling layer and what is their meaning?

How can we compute the dimensionality of the output of a convolutional layer

Describe basic properties of AlexNet and VCG

What is the main idea of ResNet to make it very deep?
