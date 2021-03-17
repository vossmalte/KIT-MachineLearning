---
title: "Maschinelles Lernen - Grundlagen"
subtitle: "Zusammenfassung"
author: "Malle \\& Valle"
date: "\\today"
titlepage: true
titlepage-background: "background1.pdf"
#titlepage-rule-color: "009669"
#logo: "kit-logo-de-rgb.pdf"
toc-own-page: true
---

## Introduction

Zusammenfassung der Vorlesung _Maschinelles Lernen -- Grundverfahren_ von Prof. Gerhard Neumann im Wintersemester 2020/2021 am Karlsruher Institut für Technologie.

## Basics

- Supervised Learning: Training data includes targets / labels
  - Regression: Learn continuos function
  - Classification: Learn (discrete) class labels
- Unsupervised Learning: Training data does _not_ include labels
  - Model the data
  - Clustering (k-means)
  - Dimensionality reduction (PCA)

### Linear Algebra

- inner / outer product
- Matrix calculus: $\nabla_x f(x) := \frac{\partial f(x)}{\partial x}$
  - $\nabla_x x^T x = 2x$
  - $\nabla_x x^T A x = 2 A x$

### Probability Theory

- random variable $X$, $p(x)$ is the probability of $X$ to be $x$
- $p(x)$ is the density function
- joint distribution $p(x,y)$: probability of $x$ _and_ $y$
- conditional distribution $p(x|y)$: probability of x given y
- sum rule: $p(x) = \sum_y p(x,y)$
- chain rule: $p(x,y)$ = p(x|y) p(y)
- Bayes rule: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$ _very important_
- Expectation of a function $f$ with distribution $p$: $\mathbb{E}_p(f(x)) = \int p(x)f(x)dx$
- Monte-carlo estimation: $\mathbb{E}_p (f(x)) = 1/N \sum_{x_i \sim p(x)} f(x_i)$ _approximate by samples_
- Distributions: Bernoulli, Gaussian

### Maximum Likelihood Estimation

- find parameter $\theta$ for $p_\theta$
- Data $x$ given with labels $y$
- Fitness of $p_\theta$: $\prod_i p_\theta(x_i,y_i)$ _higher = better_
- product is hard to differentiate, take $\log$ to make it a sum
- also works for conditional distributions (supervised learning)
- MLE $\Leftrightarrow$ SSE for conditional gaussian models with constant noise
  
## Linear Regression

- fit a line: $y = f(x) + \epsilon = w_0 + w_1 x + \epsilon$ with $\epsilon \sim N(0,1)$ being Gaussian noise
- objective: minimize sum of squared error: $\text{SSE} = \sum_i^N (y_i - f(x_i))^2$
- this estimates the mean of the target function: $f(x) = \mathbb{E}(y | x)$
- Matrix form: $\text{SSE} = (y - X w)^T (y - X w)$
- $\nabla_w \text{SSE} = 0$ yields a closed form solution: $w^* = (X^T X)^{-1}X^T X y$
- solution available because quadratic in weights (convex, differentiable)

### Quality of the model

- R-square determines how much of the total variation in $y$ is explained by the variation in $x$

### Generalized Linear Regression

- Polynomial curve fitting: $f(x) = \Phi(x)^T w$ with $\Phi(x)^T = (1, x, x^2, x^3)$
- there is a closed form solution for optimal $w^*$
- any linear closed form solution can be learned if we know suitable basis function (which we don't)

### Polynomial overfitting

- high order $\Rightarrow$ Overfitting (strong oscillation)
- error on the _training set_ is not an indication for a good fit $\Rightarrow$ need independent _test set_

### Regularization

- Goal: limit the model to not fit the training data perfectly anymore
- simple approach: force weights to be small
  - cost function: data term + regularization term, $E_D(w) + \lambda E_W(w)$
  - $\lambda$ is _regularization factor_, often needs manual tuning (strong underfitting / no effect possible)
  - Ridge regression (for SSE): $\lambda E_W(w) = \lambda w^t w$
    - optimal solution: $w^*_\text{ridge} = \Phi^T \Phi + \lambda I)^{-1}\Phi^T y$

## Linear Classification

- Generative Models (c$\mapsto$x)
  - assume class prior $p(c)$ and class density $p(x|c)$
  - predict by computing posterior $p(c|x)$ with Bayes rule
  - assumptions, e.g. Gaussian, can introduce big errors
- Discriminative Models (c$\mapsto$x)
  - estimate parameters of $p(c|x)$ from training data
  - simpler than generative models

- Binary classifier $f(x_i) = \begin{cases}>0 &\text{if } y_i = 1 \\ <0 & \text{if } y_i = 0\end{cases}$
- Linear classifier: $f(x) = w^Tx+b$
  - $w$ is normal to the discrimative hyperplane, $b$ is bias
- not all data is linear separable (e.g. two moons)

### Loss functions

- 01-loss (step function) is NP-hard to optimize
- SSE is not robust to outliers
- sigmoid $\sigma(x) = 1 / (1+e^{-x})$ _squishifier_
  - probabilistic view: $p(c=1|x) = \sigma(w^Tx+b)$
  - $p(c|x) = p(c=1|x)^c p(c=0|x)^{1-c}$
  - do log-likelihood with the above $\rightarrow$ _cross-entropy loss_
  - optimizing cross-entropy loss is called _logistic regression_ (is still convex)
  - no closed form solution but there is _gradient descent_

- use feature space to transform input space into a linear separable problem, _Generalized Logistic Models_

- Regularization: regularization penalty, e.g. $||w||^2$

### Gradient Descent

- finds a minimum (not necessarily the global minimum unless convex)
- $x_{t+1} = x_t - \eta \nabla f(x_t)$, $\eta$ is the the learning rate, gradient points in the direction of steepest ascent
- choose a good $\eta$, or the algorithm will not converge or be very slow
- when to terminate: small gradient, small change in function value, fixed number of steps (time, budget)

- _stochastic gradient descent_: don't use all training data, just a random sample
  - struggles to find the exact optimum (drunk man down the hill)
  - batch gradients require more (redundant computation)
  - mini-batches are good for GPU computation

- diminishing step size, e.g. $\eta_t = 1/t$
  - if $\sum_t \eta_t$ does not converge but $\sum_t \eta_t^2$ does

### Multi-class Classification

- softmax: $p(c=i|x) = \frac{\exp(w_i^T\Phi(x))}{\sum_k \exp(w_k^T\Phi(x))}$
  - each class gets a weight vector
  - if only two classes choose sigmoid
- as conditional multinomial distribution: $p(c|x) = \prod_k p(c=k|x)^{h_{c,k}}$
  - use log-likelihood and gradient descent

## Model Selection

- true (random test point) risk vs. empirical (test set) risk
- too complex model $\Rightarrow$ overfitting
  - unexpected behavior in between training points
  - training error goes down, validation error goes up
- too simple model $\Rightarrow$ underfitting
  - both training and validation error high
- true risk is u-shaped over the complexity
- Bias-Variance Decomposition
  - Expected Loss = Variance + Bias² + Noise
  - Bias due to restriction of model class (_structure error_)
  - Variance due to randomness of the data set
  - ![ ](bias-variance-decomposition.png)

### Evaluation Methods

- Hold-out method
  - split in training and validation
  - choose model with best performance on validation
  - "unlucky" splits cause misleading results
- k-fold cross validation
  - creates k predictors for each model class
  - Leave-One-Out: k = size of the data set (validate with 1 sample)
  - random sub-sampling: just take some samples for validation

### Regularization techniques

- Occam's Razor: prefer simpler explanations over complex ones
- Limit model complexity (neurons, leaves, polynomial order, ...)
- regularization penalty
  - small parameters $\Rightarrow$ smoother function estimate
  - $L_2, L_1$ penalty
- early stopping
  - do not push the model too much
  - use validation error to decide when to stop
  - efficient / simple but needs validation data
  - similar effect as $L_2$
- Noise
  - add noise to the inputs $\Rightarrow$ rules out non-robust solutions
  - equivalent to $L_2$ for linear regression
- Data Augmentation
  - create additional samples (flip, rotate, crop, ...)

## Trees and Forests

### k-Nearest-Neighbor

- kNN is a non-parametric (_instance-based_) methods, i.e. stores all training data and use these for predictions
  - fast / no training
  - complexity adapts to training input
  - slow prediction
  - bad at high-dim input
- kNN: find k nearest neighbors and return "average" of these
- decision boundary ~ Voronoi diagram
- choose k by cross-validation (small k $\Rightarrow$ overfitting)
- metrics: euclidean, cosine, hamming, manhattan, Mahalanobis
- performance of kNN degrades with more (irrelevant) dimensions
- curse of dimensionality
  - in high dimensions most points are far away from the origin
  - most points are equally far away from each other $\Rightarrow$ neighborhood is hard to calculate
  - remedies: feature selection, dimensionality reduction
- KD-trees to find the neighbors
  - choose axis, split at median, repeat until a

## Dimensionality Reduction

## Density Estimation

## Expectation Maximization

## Kernel Methods

## Support Vector Machines

## Bayesian Learning

## Neural Networks

## Conclusion

Thank you, `pandoc`!
