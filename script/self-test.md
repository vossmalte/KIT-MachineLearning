---
title: "Maschinelles Lernen - Grundlagen"
subtitle: "Self-test questions"
author: "Malle \\& Valle"
date: "\\today"
---

## Linear Regression

What a regression problem is

How to obtain the Least Squares solution in closed form
- Only possible as the cost function is quadratic in the weights

Generalized Linear Regression
- Non linear functions in x are fine as long as linear in w

Avoid overfitting by keeping the weights small

Expectations can be evaluated by samples

How to compute the ML estimator

Maximum likelihood is equivalent to minimizing the squared loss for:
- Conditional Gaussian models with constant noise

## Linear classification

Relation between maximum likelihood and least squares

What is a linear classification problem ...

... and how to formalize it as likelihood maximization problem
- Sigmoid likelihood for binary classification
- Soft max likelihood for multi class

What is gradient descent, stochastic gradient descent and mini batches?

How to apply gradient descent to logistic regression

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
    - similarity measure
    - provide more powerful representation (implicit feature mapping)
What do we mean by the “Kernel trick”?
    - get a more powerful representation than standard linear feature models: map data to inf. dim. feature space; use it by evaluating inner product of feature vectors
How to use kernels in regression (using Kernel Regression)?
    - evaluate w* = Phi^T * (K + \lambda I)^{-1} * y => need to invert Kernel matrix. (otherwise would need to invert covariance matrix)
How to use kernels in classification (using SVMs)?
    - w^* = \sum lambda_i y_i phi(x_i) (using constraint optimisation)
Understand how to obtain dual optimization problems from the primal
    - primal: argmin_x f(x): h(x)> b. 
    - Lagrangian: f(x) - \lambda (h(x) - b)
    - dual: max_\lambda: min_x (Lagrangian)
    => they are equivalent if we have convex constraints and a convex f
... and its relation to kernel methods
    - used for SVMs
What is the definition of a kernel and its relation to an underlying feature space?
    - kernel: k is spd
    - each kernel is equiv. to a feature space (possibly inf. dim.)

## SVM

Why is it good to use a maximum margin objective for classification?
    - maximizes "confidence", rules out unlikely seperators
How can we define the margin as optimization problem?
    - distance to margin: ( w^Tx_i + b ) / ||w||
    - constraints: y_i(w^Tx_i + b) >= 1 => abstand = y_i(w^Tx_i + b)/||w|| >= 1/||w||
    - goal: maximize 2/ ||w|| so that all data points have distance at least 1/||w|| to separator
What are slack variables and how can they be used to get a “soft” margin?
    - constraint: ">= 1 - \xi_i" -> allows violation of margin, part of regularisation: minimize those violations
How is the hinge loss defined?
    - max(0, 1 - y_i(w^Tx_i + b))
What is the relation between the slack variables and the hinge loss?
    - equivalent (hinge loss variant is derived from slack variables variant)
What are the advantages and disadvantages in comparison to logistic regression?
    - more robust to ausreißer
    - more robust
What is the difference between gradients and sub gradients
    - gradient is a sub gradient
    - gradient only exists for differentiable functions
    - subgradient are not unique. s(x) valid <=> f(x + h) >= f(x) + s(x)^T*h
## Bayesian ML

What are the 2 basic steps behind Bayesian Learning?
    - compute posterior
    - predict distribution
Why is Bayesian Learning more robust against overfitting?
    - also considers uncertainity of model as part of variance, integrate over all possible parameters (treat them as variables)
What happens with the posterior if we add more data to the training set?
    - distribution gets narrower (more confidence for parameters)
What is completing the square and how does it work?
    - quadratische ergänzung
For which 2 cases can Bayesian Learning be solved in closed form?
    - kernelized regression
    - linear regression
Which approximations can we use if no closed form is available?
    - ???
How can we derive Bayesian Linear regression
    - ...
What is the advantage of Bayesian Linear regression to Ridge regression? What is
the conceptual difference?
    - also considers uncertainity of model as part of variance
What is the major advantage of GPs over Kernel Ridge Regression?
    - also considers uncertainity of model
Why are GPs a Bayesian approach?
    - can be derived from Bayesian Regression point of view
What principle allowed deriving GPs from a Bayesian regression point of view?
    - ☺
## Neural Nets

How does logistic regression relate to neural networks?
    - last layer: linear combination of features ("weight vector")
    - other layers: "feature selection"
What kind of functions can single layer neural networks learn?
    - any ☺ - if activation function nonlinear and inf. neurons availible.
Why do we need non linear activation functions?
    - multiple "linear layers" can be squashed down to one
        - we need nonlinearities to exploit mult. layers
    - to being able to approximate any function (i.e."xor")
What activation functions can we use and what are the advantages/disadvantages of those?
    - sigmoid
    - tanh
    - ReLU
    - leaky ReLU
    - ELU
What output layer and loss function to use given the task (regression,classification)?
    - output layer | loss function
        - reg: linear layer | SSE or negloglik
        - class: sigmoid | negloglik or linear | hingeloss or softmax | negloglik

Why not use a sigmoid activation function?
    - saturation kills gradient
Derive the equations for forward and backpropagation for a simple network
    - nö.
What is mini batch gradient descent? Why use it instead of SGD or full gradient descent?
    - intermediate version of SGD and full gradient descent, 
    - cheaper than full, and more reliable than SGD
Why neural networks can overfit and what are the options to prevent it?
    - because everything can overfit
    - prevent:
        - early stopping
        - regularisation
        - dropout
        - ensemble
Why is the initialization of the network important?
    - neurons should learn *different* features, must be seperated somehow -> randomness
    - activations tend to zero for deeper networks, no gradients -> use Xavier init.
What can you read from the loss curves during training (validation and training loss)?
    - high difference = overfitting
    - no difference = underfitting
    - training loss plateau: adapt learning rate (make smaller)
    - validation loss decreases slowly: make learning rate bigger
How can we accelerate gradient descent? How does Adam work?
    - momentum (interpret as velocity)
    - gradient normalisation (lear faster when it's flat)
    - adam: use both

## RNN, DNN

Why are fully connected networks for images a bad idea and why do we need images?
    - many weights
    - for autonomous driving
What are the key components of a CNN?
    - filters, pooling layers
What hyper parameters can we set for a convolutional layer and what is their
meaning?
    - number of filters, filter size, stride, padding
What hyper parameters can we set for a pooling layer and what is their meaning?
    - stride, size (size - stride: amount of overlapping, stride: size of output)
How can we compute the dimensionality of the output of a convolutional layer
    - (size - kernel_size + 2*padding)/stride + 1
Describe basic properties of AlexNet and VCG
    - deep networks
    - convolutions + pooling, then FC at end
What is the main idea of ResNet to make it very deep?
    - more layers (works using Residual blocks)
