## Overview and Recap
This chapter discusses methods to move beyond linear models. The core idea is to replace the vector of inputs **X** with additional variables denoting the transformation of X and use linear models in the new feature space.

Taylor expansion of the true function:
$f(X) = f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2! + ... + f^{k}(a)(x-a)^{k}/k!$

Linear function from the Taylor expansion:
$f(X) \approx f(a) + f'(a)(x-a)$

Quadratic function from the Taylor expansion:
$f(X) \approx f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2!$

- Advantages of linear models:
  - In regression problems with small N and/or large P, only a linear model might able to fit without overfitting. 
  - Convenient for interpretation and inference.
  - Computationally efficient.
  - First order Taylor expansion of the true function.

 

Basis function:
$h_{m}(X): R^{P} \rightarrow R$, the $m^{th}$ tranformation of X.

Linear basis expansion in X:
    $f(X) = \sum_{m=1}^{M} \beta_{m}h_{m}(X)$

**Advantages of basis expansions:**
- Once $h_{m}$ has been determined, the model is linear in new variables.
- Achieve more flexible representations for f(X).

**Methods discussed in this chapter:**
- Useful families of piecewise polynomials and splines.
- Wavelet bases for modeling signals and images.
  
How to control a very large number of basis functions:
- Restriction methods: Additive models, tree-based methods.
- Selection methods: CART, MARS (Multivariate Adaptive Regression Splines), boosting.

