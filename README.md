# Regression and Classification on the Auto and Weekly Datasets

## Overview

**Part 1 - Linear regression on `Auto`.** Predicts fuel economy (mpg)
from engine and design characteristics. Covers simple and multiple
regression, diagnostic plots, interaction terms, and non-linear
transformations.

**Part 2 - Classification on `Weekly`.** Predicts the direction of
weekly S&P 500 returns (1990–2010) using five methods: logistic
regression, LDA, QDA, Naive Bayes, and KNN. Models are trained on
1990–2008 and evaluated on 2009–2010.

## Key results

- Simple regression of mpg on horsepower: R² ≈ 0.61, with clear
  non-linearity in the residuals. A log transformation improves
  R² to ≈ 0.67.
- Multiple regression with all predictors: R² ≈ 0.82. `weight`,
  `year`, `origin`, and `displacement` are significant; `horsepower`
  loses significance due to collinearity.
- Best classifier on held-out 2009–2010 data: logistic regression
  (or equivalently LDA) on `Lag2` alone, with 62.5% test accuracy.
  More complex models (extra lags, interactions, KNN) did not improve
  on this baseline.

## Stack

Python, pandas, NumPy, statsmodels, scikit-learn, ISLP, matplotlib,
Jupyter.

## Running it

```bash
pip install -r requirements.txt
jupyter lab finalproject.ipynb
```

## Files

- `finalproject.ipynb` - the analysis notebook
- `Textbook.pdf, Weekly.csv`  - the two datasets
- `finalproject.pdf` - rendered output
