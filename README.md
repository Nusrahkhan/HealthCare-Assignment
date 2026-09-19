# Healthcare Costs: EDA, Regression & Classification

A small machine learning exercise on a 50-patient healthcare dataset. The notebook explores the data, then tries to **predict annual healthcare cost (regression)** and **predict whether a patient has a chronic condition (classification)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nusrahkhan/HealthCare-Assignment/blob/main/healthcare_ds.ipynb)

> **Headline result:** none of the models beat a naive baseline. With only 50 rows and four weak features, there is very little predictive signal in this dataset. See [Results](#results) and [Limitations](#limitations).

---

## Repository contents

| File | Description |
|---|---|
| `healthcare_ds.ipynb` | The full analysis notebook (EDA → regression → classification) |
| `healthcare_costs.csv` | The dataset (50 rows, 6 columns) |
| `README.md` | This file |

## Dataset

`healthcare_costs.csv` has 50 patients, one row each, with no duplicate IDs and no missing values.

| Column | Type | Description |
|---|---|---|
| `PatientID` | int | Unique identifier (1001–1050). Dropped before modeling. |
| `Age` | int | Patient age (21–79) |
| `BMI` | float | Body mass index (18.8–39.3) |
| `SmokingStatus` | categorical | `Smoker` (29) / `Non-Smoker` (21) |
| `ChronicCondition` | categorical | `Yes` (27) / `No` (23) |
| `AnnualCost` | float | Annual healthcare cost (≈ 5,150–17,359) |

## What the notebook does

### 1. Exploratory data analysis
- Histograms of `BMI`, `Age`, and `AnnualCost`
- Scatter plots: Age vs. cost, Age vs. BMI, BMI vs. cost
- Correlation matrix of the numeric columns
- Box plots and group statistics of cost by `ChronicCondition`, by `SmokingStatus`, and by both together
- Box plots to check `AnnualCost` and `BMI` for outliers

Key observations:
- **Age** has a weak positive correlation with cost (0.26). **BMI** is essentially uncorrelated with cost (−0.09), and Age and BMI are uncorrelated with each other.
- Mean cost is higher for patients with a chronic condition (≈ 10,612 vs. 9,458) and for smokers (≈ 10,420 vs. 9,614), but the groups overlap heavily.
- The highest-cost group is smokers with a chronic condition (mean ≈ 11,384). Non-smokers with a condition (≈ 9,490) cost about the same as patients with no condition.

### 2. Preprocessing
- Regression: `pd.get_dummies` one-hot encoding of `ChronicCondition` and `SmokingStatus`; `PatientID` dropped.
- Classification: `LabelEncoder` on `SmokingStatus`; features are `Age`, `BMI`, `SmokingStatus`, `AnnualCost`; target is `ChronicCondition`.

### 3. Regression: predict `AnnualCost`
- 80/20 train/test split (`random_state=40`)
- Models: **Linear Regression** and **Random Forest Regressor** (300 trees)
- Evaluated with MAE, RMSE, R², a predicted-vs-actual plot, and a residual plot

### 4. Classification: predict `ChronicCondition`
- Models: **Logistic Regression** and **Random Forest Classifier** (100 trees, `min_samples_leaf=3`)
- Evaluated with 5-fold cross-validated accuracy (`KFold` for logistic regression, `StratifiedKFold` for the random forest)

## Results

### Regression (single 80/20 split, 10 test patients)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 2,383 | 2,639 | −0.86 |
| Random Forest Regressor | 2,376 | 2,737 | −1.00 |

A negative R² means the model predicts **worse than always guessing the mean cost**.

### Classification (5-fold CV accuracy)

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.46 |
| Random Forest Classifier | 0.42 |
| *Baseline: always predict the majority class (`Yes`)* | *0.54* |

Both classifiers score **below** the 27/50 = 0.54 majority-class baseline.

## Limitations

- **Very small sample.** With 50 rows, a single 80/20 split leaves only 10 test patients, so the regression scores depend heavily on which 10 were drawn. Repeated cross-validation would give a more stable estimate.
- **Weak features.** The correlations and group differences in the EDA are small relative to the spread in cost (standard deviation ≈ 2,600). The dataset looks like it may be synthetic or randomly generated.
- **Possible leakage in classification.** `AnnualCost` is used as a predictor of `ChronicCondition`. If the goal is to predict a condition *before* costs are known, drop it.
- **No hyperparameter tuning or feature scaling** was applied.
  
## Setup

**Requirements:** Python 3.9+ and the following packages.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Run locally**

1. Place `healthcare_costs.csv` in the same folder as the notebook.
2. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook healthcare_ds.ipynb
   ```
3. Run all cells top to bottom.

**Run in Google Colab:** click the badge at the top, then upload `healthcare_costs.csv` to the Colab session files before running.

## Tech stack

Python · pandas · matplotlib · seaborn · scikit-learn
