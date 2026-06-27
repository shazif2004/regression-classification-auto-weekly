# STA 141A Final Project: Regression and Classification
 
This project applies linear regression and classification methods from *An Introduction to Statistical Learning* (ISL) to two classic datasets. Part 1 uses the **Auto** dataset to predict fuel efficiency. Part 2 uses the **Weekly** dataset to predict stock market direction.
 
---
 
## Datasets
 
| Dataset | Source | Rows | Task |
|---------|--------|------|------|
| `Auto.csv` | ISL / UCI | 392 (after cleaning) | Regression |
| `Weekly.csv` | ISL | 1089 | Classification |
 
---
 
## Part 1: Linear Regression (Auto Dataset)
 
**Goal:** Predict `mpg` (miles per gallon) from car characteristics.
 
### Data Cleaning
 
The `horsepower` column had some `"?"` entries. Those rows were dropped and the column was converted to float, leaving 392 usable rows.
 
### What Was Done
 
**Simple linear regression** (`mpg ~ horsepower`)
- Clear negative relationship: every extra unit of horsepower reduces mpg by about 0.16
- R-squared: 0.61, RSE: 4.9 mpg
- Predicted mpg at horsepower = 98: **24.47**
  - 95% confidence interval: (23.97, 24.96)
  - 95% prediction interval: (14.81, 34.12)
- Diagnostic plots showed a U-shaped residual pattern, meaning the true relationship is curved, not straight
**Scatterplot matrix and correlation analysis**
- `mpg` is most strongly correlated with `weight` (-0.83), `displacement` (-0.81), and `horsepower` (-0.78)
- `cylinders`, `displacement`, `horsepower`, and `weight` are all highly correlated with each other (above 0.84), which causes multicollinearity problems in multiple regression
**Multiple linear regression** (`mpg ~ all predictors except name`)
- R-squared jumped to 0.82, RSE dropped to 3.33
- Statistically significant predictors: `weight`, `year`, `origin`, `displacement`
- `horsepower`, `cylinders`, and `acceleration` were NOT significant once other variables were included (absorbed by collinear predictors)
- The `year` coefficient was +0.75, meaning fuel economy improved by about 0.75 mpg per model year
**Interaction terms**
- Added: `horsepower * weight`, `displacement * year`, `acceleration * horsepower`
- R-squared increased from 0.82 to 0.87
- All three interactions were statistically significant
- The `horsepower:weight` interaction makes physical sense: a car's horsepower effect on mpg depends on how heavy it is
**Variable transformations** (single predictor comparisons)
 
| Model | R-squared | RSE |
|-------|-----------|-----|
| `mpg ~ hp` | 0.606 | 4.906 |
| `mpg ~ sqrt(hp)` | 0.644 | 4.665 |
| `mpg ~ log(hp)` | 0.668 | 4.501 |
| `mpg ~ poly(hp, 2)` | 0.688 | 4.374 |
 
`log(horsepower)` gave the best single-predictor fit and is the recommended transformation for this relationship.
 
---
 
## Part 2: Classification (Weekly Dataset)
 
**Goal:** Predict whether the S&P 500 went `Up` or `Down` each week using lagged returns.
 
### Dataset Overview
 
- Weekly S&P 500 returns from 1990 to 2010
- Features: `Lag1` through `Lag5` (prior weekly returns), `Volume`
- Target: `Direction` (Up or Down)
- Class split: 55.6% Up, 44.4% Down
### Key Observation from EDA
 
The lag variables are nearly uncorrelated with this week's return (all correlations near zero). Trading volume grew steadily over time (correlation with `Year`: 0.84) but has almost no relationship to direction. This already hints that predicting the market from past returns is hard.
 
### Logistic Regression (Full Model)
 
Fit with all five lags plus Volume. Only **Lag2** was statistically significant (p = 0.03). The full model essentially predicted "Up" almost every week and achieved 56.1% accuracy, which is no better than always guessing "Up."
 
### Train/Test Split Experiment
 
**Training:** 1990 to 2008 | **Test:** 2009 to 2010
 
All models below used only `Lag2` as the predictor (the only significant one).
 
**Results on held-out test data:**
 
| Method | Test Accuracy |
|--------|--------------|
| Logistic Regression (Lag2) | **62.5%** |
| LDA (Lag2) | **62.5%** |
| QDA (Lag1 + Lag2) | 58.65% |
| Naive Bayes (Lag1 + Lag2) | 58.65% |
| KNN (K=20, Lag2) | 59.62% |
| KNN (K=1, Lag2) | 50.0% |
 
**Best confusion matrix (Logistic Regression with Lag2):**
 
```
              Truth
              Down   Up
Predicted
Down            9     5
Up             34    56
```
 
### Key Takeaways from Classification
 
- Logistic regression and LDA tied at 62.5%. They agree because with one predictor and approximately normal data, both methods draw nearly the same decision boundary.
- QDA and Naive Bayes predicted "Up" for every test week, so their accuracy just reflects the class imbalance.
- KNN with K=1 overfit badly at 50%. The sweet spot for KNN was around K=20 (~60%), but it still did not beat the linear models.
- Adding more lags, interactions, or extra complexity did NOT help. More predictors mostly added noise.
- The simplest model (logistic regression on Lag2 only) won across every combination tried.
---
 
## Libraries Used
 
```python
pandas
numpy
matplotlib
statsmodels
sklearn (LDA, QDA, GaussianNB, KNeighborsClassifier, StandardScaler)
ISLP (ModelSpec, summarize, poly, confusion_table)
```
 
---
 
## How to Run
 
1. Place `Auto.csv` and `Weekly.csv` in the same directory as the notebook.
2. Install dependencies:
```bash
   pip install pandas numpy matplotlib statsmodels scikit-learn ISLP
```
3. Open `STA141A_final_project_spring_26.ipynb` in Jupyter.
4. Run all cells top to bottom (`Kernel > Restart & Run All`).
---
 
## File Structure
 
```
.
├── STA141A_final_project_spring_26.ipynb   # Main notebook
├── Auto.csv                                 # Car dataset (regression)
├── Weekly.csv                               # Stock market dataset (classification)
└── README.md
```
 
---
