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
  - Classification: Learn class labels
- Unsupervised Learning: Training data does _not_ include labels
  - Model the data
  - Clustering (k-means)
  - Dimensionality reduction (PCA)

### Linear Algebra

- inner / outer product
- Matrix calculus: $\nabla_x f(x) := \frac{\partial f(x)}{\partial x}$
  - $\nabla_x x^T x = 2x$
  - $\nabla_x x^T A x = 2 A x$
  
## Linear Regression

- fit a line: $y = f(x) + \epsilon = w_0 + w_1 x + \epsilon$ with $\epsilon \sim N(0,1)$ being Gaussian noise
- objective: minimize sum of squared error: $\text{SSE} = \sum_i^N (y_i - f(x_i))^2$
- this estimates the mean of the target function: $f(x) = \mathbb{E}(y | x)$
- Matrix form: $\text{SSE} = (y - X w)^T (y - X w)$
- $\nabla_w \text{SSE} = 0$ yields a closed form solution: $w^* = (X^T X)^{-1}X^T y$
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
- prevent overfitting by forcing small weights
  - cost function: data term + regularization term, $E_D(w) + \lambda E_W(w)$
  - $\lambda$ is _regularization factor_, often needs manual tuning (strong underfitting / no effect possible)
  - Ridge regression (for SSE): $\lambda E_W(w) = \lambda w^T w$
    - optimal solution: $w^*_\text{ridge} = (\Phi^T \Phi + \lambda I)^{-1}\Phi^T y$

## Linear Classification

## Model Selection

## Trees and Forests

## Dimensionality Reduction

## Density Estimation

## Expectation Maximization

## Kernel Methods

## Support Vector Machines

## Bayesian Learning

## Neural Networks

## Conclusion

Thank you, `pandoc`!
